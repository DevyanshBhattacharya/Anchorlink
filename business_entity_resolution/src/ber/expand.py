"""2-hop S2 <-> S3 expansion (Module 3.4, step 2).

A Malayalam-named S3 record is often far from its S1 in every lexical view, but
close to its Latin-named S2 twin, which the S1 *did* retrieve.  So for each of an
S1's strongest candidates, add that candidate's nearest same-country neighbours
**from the other source**.

The neighbour lists are built once per country from the same sparse indexes
blocking already uses, so the extra cost is one more pass over the pool rather
than a new model.
"""
from __future__ import annotations

import gc
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .blocking import SparseIndex, make_view


def build_cross_source_neighbours(index_other: SparseIndex, names: Sequence[str],
                                  addrs: Sequence[str], cfg, k: int = 3,
                                  chunk: int = 20_000) -> Tuple[np.ndarray, np.ndarray]:
    """For every record of one source, its top-k neighbours in the *other* source."""
    n = len(names)
    out_i = np.full((n, k), -1, dtype=np.int32)
    out_s = np.zeros((n, k), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        texts = [make_view(cfg, nm, ad) for nm, ad in zip(names[lo:hi], addrs[lo:hi])]
        i, s = index_other.query(texts, k=k)
        out_i[lo:hi], out_s[lo:hi] = i, s
    return out_i, out_s


def expand_two_hop(cand: pd.DataFrame, neighbours: Dict[int, Tuple[np.ndarray, np.ndarray]],
                   pool_pos: Dict[int, Dict[int, int]],
                   pool_ids: Dict[int, np.ndarray],
                   top_seeds: int = 5, per_seed: int = 3,
                   score_col: str = "rrf") -> pd.DataFrame:
    """Add, for each S1's top seeds, their nearest other-source neighbours.

    ``neighbours[src]`` is ``(idx, score)`` for records of source ``src``, giving
    rows of the *other* source's pool.  Returns the new rows only; the caller
    unions them with the existing candidate table.
    """
    if cand.empty:
        return cand.iloc[:0].copy()
    order = np.lexsort((-cand[score_col].to_numpy(), cand["s1"].to_numpy()))
    s1 = cand["s1"].to_numpy()[order]
    src = cand["src"].to_numpy()[order]
    cid = cand["cid"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    bounds = np.r_[starts, len(s1)]

    new_s1: List[int] = []
    new_src: List[int] = []
    new_cid: List[int] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seen = set(zip(src[a:b].tolist(), cid[a:b].tolist()))
        for j in range(a, min(a + top_seeds, b)):
            s = int(src[j])
            other = 3 if s == 2 else 2
            nb = neighbours.get(s)
            if nb is None:
                continue
            row = pool_pos[s].get(int(cid[j]))
            if row is None:
                continue
            idx, _ = nb
            for t in idx[row, :per_seed]:
                if t < 0:
                    continue
                oid = int(pool_ids[other][t])
                if (other, oid) in seen:
                    continue
                seen.add((other, oid))
                new_s1.append(int(s1[a]))
                new_src.append(other)
                new_cid.append(oid)
    out = pd.DataFrame({"s1": np.asarray(new_s1, dtype=np.int32),
                        "src": np.asarray(new_src, dtype=np.int8),
                        "cid": np.asarray(new_cid, dtype=np.int32)})
    for col in cand.columns:
        if col not in out.columns:
            out[col] = np.float32(0) if cand[col].dtype.kind == "f" else np.int16(999) \
                if col.endswith("_rank") else cand[col].dtype.type(0)
    return out[cand.columns]
