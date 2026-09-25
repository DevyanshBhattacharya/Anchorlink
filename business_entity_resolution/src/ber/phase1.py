"""Phase 1 driver: prove the dense retriever and measure what it adds.

Full-scale dense retrieval is AWS work — encoding ~22M records with BGE-M3 and
searching 4.7M-vector partitions does not fit this machine.  What *does* fit, and
is what this module does, is the honest small-scale version:

* build a self-contained **slice** — a few thousand S1 clusters plus a pool of
  their true matches and a realistic mass of distractors;
* measure the sparse union's recall on that slice, so there is a baseline;
* measure a dense retriever on the same slice, zero-shot and after fine-tuning;
* report the union of the two.

Every number is therefore comparable within the slice, and the code that runs
here is the same code that runs at full scale with a different config.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from . import config
from .data import load_partition
from .dense import Encoder, EncoderConfig, dense_topk, serialise
from .device import pick_device
from .io_utils import read_ground_truth
from .splits import build_name_families, fold_ids, sample_by_family


@dataclass
class Slice:
    """A self-contained retrieval problem: queries, a pool, and the truth."""
    country: str
    q_ids: List[str]
    q_name: List[str]
    q_addr: List[str]
    p_ids: List[str]
    p_name: List[str]
    p_addr: List[str]
    truth: Dict[str, Set[str]]

    def stats(self) -> Dict[str, float]:
        n_true = sum(len(v) for v in self.truth.values())
        return {"queries": len(self.q_ids), "pool": len(self.p_ids),
                "true_pairs": n_true,
                "avg_true": n_true / max(len(self.q_ids), 1),
                "singletons": sum(1 for v in self.truth.values() if not v)}


def build_slice(country: str, n_clusters: int = 8000, pool_size: int = 120_000,
                fold: int = 0, seed: int = config.SEED, split: str = "train") -> Slice:
    """Queries from one fold, their true matches, and distractors up to ``pool_size``."""
    rng = np.random.default_rng(seed)
    fam = build_name_families(split)
    mask = (fam["country"].to_numpy() == country) & (fam["fold"].to_numpy() == fold)
    q_ids_num = sample_by_family(fam, mask, n_clusters, seed)
    gt = read_ground_truth()

    s1 = load_partition(split, "S1", country)
    pos1 = {int(i): j for j, i in enumerate(s1.ids)}
    q_ids, q_name, q_addr, truth = [], [], [], {}
    wanted: Dict[int, Set[int]] = {2: set(), 3: set()}
    for i in q_ids_num:
        j = pos1.get(int(i))
        if j is None:
            continue
        key = f"S1-{int(i)}"
        q_ids.append(key)
        q_name.append(s1.names[j])
        q_addr.append(s1.addrs[j])
        t = set(gt.get(key, ()))
        truth[key] = t
        for m in t:
            wanted[2 if m[1] == "2" else 3].add(int(m[3:]))
    del s1, pos1, gt
    gc.collect()

    p_ids, p_name, p_addr = [], [], []
    per_source_extra = max(0, (pool_size - sum(len(v) for v in wanted.values()))) // 2
    for src in (2, 3):
        p = load_partition(split, f"S{src}", country)
        ids = p.ids
        want = np.fromiter(wanted[src], dtype=np.int32)
        is_true = np.isin(ids, want)
        rows = list(np.flatnonzero(is_true))
        others = np.flatnonzero(~is_true)
        take = min(per_source_extra, len(others))
        if take:
            rows += list(rng.choice(others, size=take, replace=False))
        for r in rows:
            p_ids.append(f"S{src}-{int(ids[r])}")
            p_name.append(p.names[r])
            p_addr.append(p.addrs[r])
        del p
        gc.collect()
    return Slice(country, q_ids, q_name, q_addr, p_ids, p_name, p_addr, truth)


# ------------------------------------------------------------------ metrics
def recall_at_k(cand: Dict[str, Set[str]], truth: Dict[str, Set[str]]) -> Dict[str, float]:
    hit = tot = full = ns = 0
    n_cand = 0
    for q, t in truth.items():
        c = cand.get(q, set())
        n_cand += len(c)
        hit += len(c & t)
        tot += len(t)
        if t:
            ns += 1
            full += int(t <= c)
    return {"pair_recall": hit / tot if tot else float("nan"),
            "full_cluster_recall": full / ns if ns else float("nan"),
            "avg_candidates": n_cand / max(len(truth), 1)}


def sparse_slice_candidates(sl: Slice, bcfg, k: int) -> Dict[str, Set[str]]:
    """Run the production retriever views over the slice pool."""
    from .blocking import SparseIndex, make_view
    out: Dict[str, Set[str]] = {q: set() for q in sl.q_ids}
    for tag, cfg in bcfg.specs():
        idx = SparseIndex(cfg).build([make_view(cfg, n, a)
                                      for n, a in zip(sl.p_name, sl.p_addr)])
        I, _ = idx.query([make_view(cfg, n, a) for n, a in zip(sl.q_name, sl.q_addr)], k=k)
        for r, q in enumerate(sl.q_ids):
            out[q].update(sl.p_ids[c] for c in I[r] if c >= 0)
        del idx
        gc.collect()
    return out


def dense_slice_candidates(enc: Encoder, sl: Slice, k: int,
                           batch_size: int = 64) -> Tuple[Dict[str, Set[str]], Dict]:
    t0 = time.time()
    pool_vecs = enc.encode([serialise(n, a, sl.country)
                            for n, a in zip(sl.p_name, sl.p_addr)], batch_size)
    t_pool = time.time() - t0
    t0 = time.time()
    q_vecs = enc.encode([serialise(n, a, sl.country)
                         for n, a in zip(sl.q_name, sl.q_addr)], batch_size)
    t_q = time.time() - t0
    I, S = dense_topk(q_vecs, pool_vecs, k)
    out = {q: {sl.p_ids[c] for c in I[r] if c >= 0} for r, q in enumerate(sl.q_ids)}
    timing = {"encode_pool_s": round(t_pool, 1), "encode_query_s": round(t_q, 1),
              "records_per_s": round(len(sl.p_ids) / max(t_pool, 1e-9), 1),
              "dim": int(pool_vecs.shape[1])}
    del pool_vecs, q_vecs
    gc.collect()
    return out, timing


def union(*cands: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for c in cands:
        for q, s in c.items():
            out.setdefault(q, set()).update(s)
    return out
