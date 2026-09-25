"""Phase 1 — the dense bi-encoder: losses, batching, encoding and ANN search.

Why a dense retriever at all: the sparse union already reaches 0.947 pair recall
on the India partition, and its misses are concentrated where character overlap
fails — native-script names, domain-style names and heavily truncated addresses.
Those are exactly what a multilingual encoder reads.

Three choices follow the research doc and matter more than the architecture:

* **Supervised contrastive loss, not InfoNCE.**  InfoNCE and Multiple Negatives
  Ranking Loss assume one positive per anchor.  Clusters here hold 3–6 records,
  so same-cluster records inside a batch would be scored as false negatives.
  SupCon (Khosla et al., NeurIPS 2020) allows many positives per anchor.
* **Collision batches.**  Batches are packed with clusters the sparse index
  confuses with each other — same name, same street, neighbouring number — so the
  hardest negatives arrive as in-batch negatives at no extra encoding cost
  (TAS-Balanced, Hofstätter et al., SIGIR 2021).  Batches never mix countries.
* **Positive-aware negative filtering.**  A mined negative scoring above
  ``max_ratio`` of the anchor's weakest positive is dropped (NV-Retriever's
  TopK-PercPos), which guards against residual label noise.

Drift control, because France has no labels: a low learning rate, one or two
epochs, and early stopping on the leave-one-country-out fold rather than the
in-country fold.
"""
from __future__ import annotations

import gc
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from . import config
from .device import check_model, pick_device
from .normalize import clean_text, house_numbers, skeleton


# ------------------------------------------------------------------ text
def serialise(name: str, addr: str, country: str) -> str:
    """One record as the encoder reads it.

    Fixed field order with explicit markers, an extracted ``nums:`` field so the
    model sees house numbers as short adjacent tokens rather than digits buried
    in a long address, and a romanised echo for native scripts.
    """
    folded = clean_text(name)
    raw = (name or "").strip()
    echo = f" ({folded})" if folded and folded != raw.lower() else ""
    nums = " ".join(sorted(house_numbers(addr))) or "-"
    return (f"name: {raw}{echo} | nums: {nums} | "
            f"addr: {(addr or '').strip() or '-'} | country: {country}")


# ------------------------------------------------------------------ losses
def supcon_loss(emb, cluster_ids, tau: float = 0.05):
    """Supervised contrastive loss; rows sharing a cluster id are positives.

    ``emb`` is ``[B, d]`` and L2-normalised.  Rows whose cluster has no other
    member in the batch (singletons) contribute as negatives only.
    """
    import torch
    B = emb.size(0)
    self_mask = torch.eye(B, dtype=torch.bool, device=emb.device)
    sim = (emb @ emb.T / tau).masked_fill(self_mask, float("-inf"))
    pos = (cluster_ids[:, None] == cluster_ids[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    keep = n_pos > 0
    if not bool(keep.any()):
        return emb.sum() * 0.0        # not sim: its diagonal is -inf
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(1)[keep] / n_pos[keep])
    return per_anchor.mean()


def collision_batches(clusters: Dict[int, Sequence[int]],
                      collisions: Dict[int, Sequence[int]],
                      country: Dict[int, str],
                      clusters_per_batch: int = 256,
                      seed: int = 0) -> Iterator[List[Tuple[int, int]]]:
    """Batches packed with clusters the sparse index confuses with each other.

    ``clusters`` maps a cluster id to its record ids, ``collisions`` maps a
    cluster id to the cluster ids it collides with.  Every cluster is emitted
    exactly once and no batch mixes countries.
    """
    rnd = random.Random(seed)
    by_country: Dict[str, List[int]] = defaultdict(list)
    used: set = set()
    for cid in clusters:
        by_country[country[cid]].append(cid)
    for ctry, cids in by_country.items():
        cids = list(cids)
        rnd.shuffle(cids)
        pool = iter(cids)
        for seed_cid in cids:
            if seed_cid in used:
                continue
            batch: List[int] = []
            frontier: List[int] = [seed_cid]
            while len(batch) < clusters_per_batch:
                cid = frontier.pop() if frontier else next(pool, None)
                if cid is None:
                    break
                if cid in used:
                    continue
                used.add(cid)
                batch.append(cid)
                frontier.extend(c for c in list(collisions.get(cid, ()))[:4]
                                if c not in used and country.get(c) == ctry)
            if batch:
                yield [(rid, cid) for cid in batch for rid in clusters[cid]]


def filter_mined_negatives(neg_scores: np.ndarray, weakest_positive: np.ndarray,
                           max_ratio: float = 0.95) -> np.ndarray:
    """NV-Retriever TopK-PercPos: drop negatives too close to the true positive."""
    return np.asarray(neg_scores) < max_ratio * np.asarray(weakest_positive)


# ------------------------------------------------------------------ encoder
@dataclass
class EncoderConfig:
    repo: str = "BAAI/bge-m3"
    max_length: int = 96
    batch_size: int = 64
    pooling: str = "cls"          # bge-m3 uses CLS
    projection_dim: int = 0       # 0 = keep the backbone dimension
    device: str | None = None
    fp16: bool = True


class Encoder:
    """A thin wrapper: load, encode to L2-normalised float32, save, load."""

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        check_model(self.cfg.repo)
        self.device = self.cfg.device or pick_device()
        self.model = None
        self.tokenizer = None
        self.projection = None

    def load(self, weights: Path | None = None) -> "Encoder":
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.repo)
        self.model = AutoModel.from_pretrained(self.cfg.repo)
        self.model.to(self.device)
        if self.cfg.projection_dim:
            d = self.model.config.hidden_size
            self.projection = torch.nn.Linear(d, self.cfg.projection_dim, bias=False)
            torch.nn.init.eye_(self.projection.weight[:min(d, self.cfg.projection_dim)])
            self.projection.to(self.device)
        if weights is not None and Path(weights).exists():
            state = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(state["model"])
            if self.projection is not None and state.get("projection") is not None:
                self.projection.load_state_dict(state["projection"])
        return self

    def _pool(self, out, mask):
        import torch
        h = out.last_hidden_state
        if self.cfg.pooling == "cls":
            return h[:, 0]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, texts: Sequence[str]):
        import torch
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.cfg.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        vec = self._pool(out, enc["attention_mask"])
        if self.projection is not None:
            vec = self.projection(vec)
        return torch.nn.functional.normalize(vec, p=2, dim=1)

    def encode(self, texts: Sequence[str], batch_size: int | None = None,
               show_progress: bool = False) -> np.ndarray:
        import torch
        bs = batch_size or self.cfg.batch_size
        self.model.eval()
        outs: List[np.ndarray] = []
        rng = range(0, len(texts), bs)
        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc="encode")
            except ImportError:
                pass
        with torch.no_grad():
            for lo in rng:
                vec = self.forward(texts[lo:lo + bs])
                outs.append(vec.float().cpu().numpy())
        return np.vstack(outs).astype(np.float32) if outs else np.zeros((0, 1), np.float32)

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "projection": None if self.projection is None
                    else self.projection.state_dict(),
                    "cfg": self.cfg.__dict__}, path)


# ------------------------------------------------------------------ search
#: faiss is opt-in.  On macOS, faiss-cpu and torch each bring their own OpenMP
#: runtime and loading both aborts the process, so the default here is an exact
#: chunked numpy search that gives identical results.  Set ``BER_USE_FAISS=1`` on
#: Linux/AWS, where faiss is both safe and much faster on a multi-million-vector
#: pool.
def _faiss_enabled() -> bool:
    import os
    return os.environ.get("BER_USE_FAISS", "0") == "1"


def dense_topk(query_vecs: np.ndarray, pool_vecs: np.ndarray, k: int,
               chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k, chunked so a large pool never needs a dense score matrix."""
    n, d = query_vecs.shape
    k = min(k, len(pool_vecs)) if len(pool_vecs) else 0
    if k == 0:
        return (np.full((n, 0), -1, np.int32), np.zeros((n, 0), np.float32))
    if _faiss_enabled():
        import faiss
        index = faiss.IndexFlatIP(d)
        index.add(np.ascontiguousarray(pool_vecs, dtype=np.float32))
        sims, ids = index.search(np.ascontiguousarray(query_vecs, dtype=np.float32), k)
        return ids.astype(np.int32), sims.astype(np.float32)
    out_i = np.empty((n, k), dtype=np.int32)
    out_s = np.empty((n, k), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        S = query_vecs[lo:hi] @ pool_vecs.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(hi - lo)[:, None]
        vals = S[rows, idx]
        order = np.argsort(-vals, axis=1)
        out_i[lo:hi] = idx[rows, order]
        out_s[lo:hi] = vals[rows, order]
    return out_i, out_s
