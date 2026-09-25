"""Phase 1 — fine-tuning the bi-encoder with SupCon on collision batches.

A training cluster is one S1 record plus its S2/S3 matches.  Every record is an
anchor and every other member of its cluster is a positive.  Records that match
nothing — 5.6% of S1 entities and about a quarter of S2/S3 records — stay in as
singleton clusters so the model learns to push distractors away.

Collisions come from the sparse blocker that already ran: if cluster A's S1
retrieved a record belonging to cluster B, the two collide.  Batching around
those collisions puts the hardest negatives in the batch for free.

Sizing: ``TrainConfig`` defaults are for a single 24 GB GPU.  The same script
runs on Apple MPS or CPU at a smaller batch size, which is how the code is proven
and the gain measured before the full run moves to AWS.
"""
from __future__ import annotations

import gc
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import config
from .data import load_partition
from .dense import Encoder, EncoderConfig, collision_batches, serialise, supcon_loss
from .device import pick_device


@dataclass
class TrainConfig:
    """Defaults are sized for one 24 GB GPU (A10G / L4 / 3090 class)."""
    repo: str = "BAAI/bge-m3"
    max_length: int = 96
    clusters_per_batch: int = 96          # ~350 records at 3.6 records per cluster
    micro_batch: int = 64                 # forward chunk; GradCache-style accumulation
    lr: float = 1e-5                      # low: keep the encoder near its pretrained weights
    epochs: int = 1
    tau: float = 0.05
    warmup_frac: float = 0.05
    max_steps: int = 0                    # 0 = no cap
    eval_every: int = 200
    patience: int = 3
    seed: int = config.SEED
    projection_dim: int = 0
    grad_clip: float = 1.0

    @classmethod
    def for_device(cls, device: str | None = None, **kw) -> "TrainConfig":
        """Shrink the batch on MPS/CPU so the same command runs anywhere."""
        device = device or pick_device()
        if device == "cuda":
            return cls(**kw)
        small = dict(clusters_per_batch=16, micro_batch=16, max_length=64)
        small.update(kw)
        return cls(**small)


# ------------------------------------------------------------------ data
@dataclass
class ClusterSet:
    """Records, their cluster ids and countries — the input the sampler needs."""
    texts: List[str]
    cluster_of: np.ndarray                 # [n_records] cluster id per record
    clusters: Dict[int, List[int]]
    country: Dict[int, str]
    record_key: List[Tuple[int, int]]      # (src, id); src 1/2/3

    def __len__(self) -> int:
        return len(self.texts)


def build_cluster_set(split: str, countries: Sequence[str], gt_dict: Dict[str, List[str]],
                      s1_ids_by_country: Dict[str, np.ndarray],
                      max_distractors_per_country: int = 0,
                      seed: int = config.SEED) -> ClusterSet:
    """Assemble the training records: every selected S1 cluster, plus distractors."""
    rng = np.random.default_rng(seed)
    texts: List[str] = []
    cluster_of: List[int] = []
    clusters: Dict[int, List[int]] = defaultdict(list)
    country: Dict[int, str] = {}
    record_key: List[Tuple[int, int]] = []
    next_cluster = 0

    for c in countries:
        want = set(int(i) for i in s1_ids_by_country[c])
        s1 = load_partition(split, "S1", c)
        pools = {2: load_partition(split, "S2", c), 3: load_partition(split, "S3", c)}
        pool_pos = {s: {int(i): j for j, i in enumerate(p.ids)} for s, p in pools.items()}
        used: Dict[int, Set[int]] = {2: set(), 3: set()}

        for j, sid in enumerate(s1.ids):
            sid = int(sid)
            if sid not in want:
                continue
            cid = next_cluster
            next_cluster += 1
            country[cid] = c
            texts.append(serialise(s1.names[j], s1.addrs[j], c))
            cluster_of.append(cid)
            clusters[cid].append(len(texts) - 1)
            record_key.append((1, sid))
            for mid in gt_dict.get(f"S1-{sid}", ()):
                src = 2 if mid[1] == "2" else 3
                num = int(mid[3:])
                k = pool_pos[src].get(num)
                if k is None:
                    continue
                used[src].add(num)
                p = pools[src]
                texts.append(serialise(p.names[k], p.addrs[k], c))
                cluster_of.append(cid)
                clusters[cid].append(len(texts) - 1)
                record_key.append((src, num))

        if max_distractors_per_country:
            for src in (2, 3):
                p = pools[src]
                cand = np.setdiff1d(p.ids, np.fromiter(used[src], dtype=np.int32),
                                    assume_unique=False)
                take = min(max_distractors_per_country // 2, len(cand))
                if take <= 0:
                    continue
                pick = rng.choice(len(cand), size=take, replace=False)
                pos = {int(i): j for j, i in enumerate(p.ids)}
                for idx in pick:
                    k = pos[int(cand[idx])]
                    cid = next_cluster
                    next_cluster += 1
                    country[cid] = c
                    texts.append(serialise(p.names[k], p.addrs[k], c))
                    cluster_of.append(cid)
                    clusters[cid].append(len(texts) - 1)
                    record_key.append((src, int(cand[idx])))
        del s1, pools, pool_pos
        gc.collect()

    return ClusterSet(texts, np.asarray(cluster_of, dtype=np.int64),
                      dict(clusters), country, record_key)


def collisions_from_candidates(cand: pd.DataFrame, cluster_of_s1: Dict[int, int],
                               cluster_of_record: Dict[Tuple[int, int], int],
                               max_per_cluster: int = 8) -> Dict[int, List[int]]:
    """Clusters the sparse blocker confuses with each other.

    If S1 ``q`` retrieved a record that belongs to cluster ``B``, then ``q``'s
    cluster and ``B`` collide.  These are the same-name / same-street /
    neighbouring-number pairs that decide the score.
    """
    out: Dict[int, Set[int]] = defaultdict(set)
    s1 = cand["s1"].to_numpy()
    src = cand["src"].to_numpy()
    cid = cand["cid"].to_numpy()
    for a, b, c in zip(s1, src, cid):
        ca = cluster_of_s1.get(int(a))
        cb = cluster_of_record.get((int(b), int(c)))
        if ca is None or cb is None or ca == cb:
            continue
        if len(out[ca]) < max_per_cluster:
            out[ca].add(cb)
        if len(out[cb]) < max_per_cluster:
            out[cb].add(ca)
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------ training
def train_bi_encoder(data: ClusterSet, collisions: Dict[int, List[int]],
                     cfg: TrainConfig | None = None,
                     eval_fn=None, out_path: Path | None = None,
                     verbose: bool = True) -> Tuple[Encoder, Dict]:
    """Fine-tune with SupCon on collision batches.  Returns (encoder, history)."""
    import torch

    cfg = cfg or TrainConfig.for_device()
    device = pick_device()
    enc = Encoder(EncoderConfig(repo=cfg.repo, max_length=cfg.max_length,
                                batch_size=cfg.micro_batch,
                                projection_dim=cfg.projection_dim,
                                device=device)).load()
    params = list(enc.model.parameters())
    if enc.projection is not None:
        params += list(enc.projection.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)

    batches = [b for b in collision_batches(data.clusters, collisions, data.country,
                                            cfg.clusters_per_batch, cfg.seed)]
    total = len(batches) * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(total, 1),
        pct_start=cfg.warmup_frac, anneal_strategy="linear")

    history: Dict = {"config": asdict(cfg), "device": device, "steps": [],
                     "n_records": len(data), "n_clusters": len(data.clusters),
                     "n_batches": len(batches)}
    best, bad, step = -1.0, 0, 0
    t0 = time.time()
    enc.model.train()
    stop = False
    for epoch in range(cfg.epochs):
        if stop:
            break
        for batch in batches:
            if cfg.max_steps and step >= cfg.max_steps:
                stop = True
                break
            texts = [data.texts[r] for r, _ in batch]
            cids = torch.tensor([c for _, c in batch], device=device)
            opt.zero_grad(set_to_none=True)
            # one forward over the whole batch; micro-batching keeps memory flat
            # for the tokenizer/attention while the loss still sees every pair
            vecs = []
            for lo in range(0, len(texts), cfg.micro_batch):
                vecs.append(enc.forward(texts[lo:lo + cfg.micro_batch]))
            emb = torch.cat(vecs, dim=0)
            loss = supcon_loss(emb, cids, tau=cfg.tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if verbose and step % 20 == 0:
                print(f"  [dense] step {step}/{total} loss={float(loss):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if eval_fn is not None and cfg.eval_every and step % cfg.eval_every == 0:
                enc.model.eval()
                score = float(eval_fn(enc))
                enc.model.train()
                history["steps"].append({"step": step, "loss": float(loss), "eval": score})
                if verbose:
                    print(f"  [dense] step {step}: eval={score:.4f} (best {best:.4f})",
                          flush=True)
                if score > best:
                    best, bad = score, 0
                    if out_path is not None:
                        enc.save(out_path)
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        if verbose:
                            print("  [dense] early stop on the cross-country fold",
                                  flush=True)
                        stop = True
                        break
    history["best_eval"] = best
    history["steps_run"] = step
    history["seconds"] = round(time.time() - t0, 1)
    if out_path is not None and best < 0:
        enc.save(out_path)
    return enc, history
