"""Probe C — does a fifth retriever earn a re-block?

The standard benchmark of this project, the one every row of `RESULTS.md` uses:
**4,000 held-out fold-0 India queries searching the whole 4,133,346-record
S2+S3 training pool**, top-30 per source per retriever.  Anything that claims to
lift the candidate ceiling has to show it here first, because turning it on costs
~2.5 h of re-blocking for the eight partitions and then a full run downstream.

Three things are compared against the production four-view union:

1. **key hits ordered by block selectivity** instead of by pool row id, with a
   larger cap — `KeyBlocks.lookup(by_selectivity=True)`, cap 150;
2. **banded MinHash (LSH)** as a fifth view — rank-free retrieval, so a true
   match sitting at rank 200 behind a crowd of look-alikes is still found;
3. both.

Imports no model, so no second OpenMP runtime reaches this process and the
sparse top-k product keeps all ten cores (PROGRESS.md §7).

Writes work/reports/probe_blocking.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, "business_entity_resolution/src")

from ber import config
from ber.blocking import make_view
from ber.corpus import build_roles
from ber.data import load_partition
from ber.keys import KeyBlocks
from ber.labels import GroundTruth
from ber.minhash import MinHashConfig, MinHashIndex
from ber.retrieval import (DEFAULT_BLOCKING, RETRIEVER_TAGS, _first_unique,
                           get_index, index_path)
from ber.splits import build_name_families, fold_ids, sample_ids

COUNTRY = "India"
SPLIT = "train"
N_QUERIES = 4_000
K = DEFAULT_BLOCKING.k_per_retriever          # 30
CAP_PER_SOURCE = DEFAULT_BLOCKING.max_union_per_source   # 80
LSH_CFG = MinHashConfig(bands=20, rows=3, max_bucket=40)
KEY_CAP_NEW = 150


def lsh_path(source: str) -> "config.Path":
    cfg = LSH_CFG
    return (config.INDEX_DIR /
            f"{SPLIT}_{COUNTRY}_{source}_{cfg.name}_b{cfg.bands}r{cfg.rows}"
            f"_m{cfg.max_bucket}_n{cfg.ngram}.pkl")


def get_lsh(source: str, names, addrs, verbose=True) -> MinHashIndex:
    p = lsh_path(source)
    if p.exists():
        if verbose:
            print(f"  [lsh] {source}: cached", flush=True)
        return MinHashIndex.load(p)
    texts = [make_view(_NA_CFG, n, a) for n, a in zip(names, addrs)]
    idx = MinHashIndex(LSH_CFG).build(texts, verbose=verbose)
    del texts
    gc.collect()
    idx.save(p)
    return idx


_NA_CFG = None   # set in main(); the name+address view, shared with the `na` index


def union_stats(hits: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                which: Sequence[str], keyhits: Dict[str, List[np.ndarray]] | None,
                pool_ids: Dict[str, np.ndarray], q_ids: np.ndarray,
                gt: GroundTruth, cap: int) -> dict:
    """Pair recall / full-cluster recall / ceiling for one combination of views.

    The union is capped per source the way production does it, by the number of
    views that found the candidate (production uses RRF; here every view in the
    combination votes equally, which is what makes the *comparison* fair between
    combinations rather than a re-tuning of the fusion).
    """
    nq = len(q_ids)
    inter = np.zeros(nq, dtype=np.int64)
    n_cand = np.zeros(nq, dtype=np.int64)
    for source in ("S2", "S3"):
        src_code = np.int8(2 if source == "S2" else 3)
        q_parts, c_parts = [], []
        for tag in which:
            I, _ = hits[source][tag]
            keep = (I >= 0).reshape(-1)
            if not keep.any():
                continue
            q_parts.append(np.repeat(np.arange(nq, dtype=np.int64), I.shape[1])[keep])
            c_parts.append(I.reshape(-1)[keep].astype(np.int64))
        if keyhits is not None:
            kh = keyhits[source]
            q_parts.append(np.concatenate([np.full(len(h), i, dtype=np.int64)
                                           for i, h in enumerate(kh) if len(h)]))
            c_parts.append(np.concatenate([h for h in kh if len(h)]).astype(np.int64))
        if not q_parts:
            continue
        q = np.concatenate(q_parts)
        c = np.concatenate(c_parts)
        del q_parts, c_parts
        n_pool = len(pool_ids[source])
        uniq, counts = np.unique(q * n_pool + c, return_counts=True)
        qq = (uniq // n_pool).astype(np.int64)
        cc = (uniq % n_pool).astype(np.int64)
        if cap > 0:
            order = np.lexsort((-counts, qq))
            g = qq[order]
            starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
            rank = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
            sel = order[rank < cap]
            qq, cc = qq[sel], cc[sel]
        y = gt.label(q_ids[qq], np.full(len(qq), src_code), pool_ids[source][cc])
        inter += np.bincount(qq, weights=y, minlength=nq).astype(np.int64)
        n_cand += np.bincount(qq, minlength=nq).astype(np.int64)
        del q, c, uniq, counts, qq, cc, y
        gc.collect()

    n_true = gt.cluster_size(q_ids).astype(np.int64)
    ns = n_true > 0
    bound = np.zeros(nq, dtype=np.float64)
    bound[~ns] = 1.0
    denom = 0.25 * n_true[ns] + inter[ns]
    bound[ns] = np.where(denom > 0, 1.25 * inter[ns] / np.maximum(denom, 1e-12), 0.0)
    return {
        "pair_recall": float(inter.sum() / max(n_true.sum(), 1)),
        "full_cluster_recall": float((inter[ns] == n_true[ns]).mean()),
        "avg_candidates": float(n_cand.mean()),
        "f05_ceiling": float(bound.mean()),
    }


def main() -> None:
    global _NA_CFG
    t0 = time.time()
    roles = build_roles(SPLIT)
    fam = build_name_families(SPLIT)
    q_ids = sample_ids(fold_ids(fam, 0, country=COUNTRY), N_QUERIES, config.SEED + 99)
    del fam
    gc.collect()

    s1 = load_partition(SPLIT, "S1", COUNTRY)
    rows = np.flatnonzero(np.isin(s1.ids, q_ids))
    q_ids = s1.ids[rows]
    q_names = [s1.names[i] for i in rows]
    q_addrs = [s1.addrs[i] for i in rows]
    del s1
    gc.collect()
    print(f"[probe] {len(q_ids):,} held-out queries, {COUNTRY} {SPLIT}", flush=True)

    bcfg = DEFAULT_BLOCKING
    specs = bcfg.specs()
    _NA_CFG = specs[0][1]                 # the name+address view
    gt = GroundTruth()

    hits: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    keyhits_old: Dict[str, List[np.ndarray]] = {}
    keyhits_new: Dict[str, List[np.ndarray]] = {}
    pool_ids: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    for source in ("S2", "S3"):
        print(f"[probe] {source}", flush=True)
        pool = load_partition(SPLIT, source, COUNTRY)
        pool_ids[source] = pool.ids.copy()
        # key blocks, both policies, from the same pool pass
        t1 = time.time()
        kb_old = KeyBlocks.build(pool.names, pool.addrs, COUNTRY, roles, max_block=50)
        kb_new = KeyBlocks.build(pool.names, pool.addrs, COUNTRY, roles,
                                 max_block=KEY_CAP_NEW)
        timings[f"{source}_keys"] = time.time() - t1
        lsh = get_lsh(source, pool.names, pool.addrs)
        del pool
        gc.collect()

        from ber.keys import record_keys
        keyhits_old[source] = []
        keyhits_new[source] = []
        for n, a in zip(q_names, q_addrs):
            ks = record_keys(n, a, COUNTRY, roles)
            h = kb_old.lookup(ks, by_selectivity=False)
            keyhits_old[source].append(np.unique(h)[:bcfg.k_keys] if len(h) else h)
            h = kb_new.lookup(ks, by_selectivity=True)
            keyhits_new[source].append(_first_unique(h, bcfg.k_keys) if len(h) else h)
        del kb_old, kb_new
        gc.collect()

        per: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tag, cfg in specs:
            idx = get_index(SPLIT, COUNTRY, source, cfg, bcfg, None, True, False)
            t1 = time.time()
            qt = [make_view(cfg, n, a) for n, a in zip(q_names, q_addrs)]
            per[tag] = idx.query(qt, k=K)
            timings[f"{source}_{tag}"] = time.time() - t1
            print(f"  [{tag}] {1000 * (time.time() - t1) / len(q_ids):.2f} s/1k",
                  flush=True)
            del idx, qt
            gc.collect()
        t1 = time.time()
        qt = [make_view(_NA_CFG, n, a) for n, a in zip(q_names, q_addrs)]
        per["lsh"] = lsh.query(qt, k=K)
        timings[f"{source}_lsh"] = time.time() - t1
        print(f"  [lsh] {1000 * (time.time() - t1) / len(q_ids):.2f} s/1k", flush=True)
        hits[source] = per
        del lsh, qt
        gc.collect()

    four = list(RETRIEVER_TAGS)
    combos = [
        ("4 views, no keys", four, None),
        ("4 views + keys (production)", four, keyhits_old),
        ("4 views + keys, selective cap 150", four, keyhits_new),
        ("4 views + LSH, no keys", four + ["lsh"], None),
        ("4 views + LSH + keys (production)", four + ["lsh"], keyhits_old),
        ("4 views + LSH + keys, selective cap 150", four + ["lsh"], keyhits_new),
        ("LSH alone", ["lsh"], None),
    ]
    out = {"queries": int(len(q_ids)), "k": K, "cap_per_source": CAP_PER_SOURCE,
           "lsh": {"bands": LSH_CFG.bands, "rows": LSH_CFG.rows,
                   "threshold": round(LSH_CFG.threshold, 4),
                   "max_bucket": LSH_CFG.max_bucket},
           "timings_s": {k: round(v, 1) for k, v in timings.items()},
           "rows": []}
    for label, which, kh in combos:
        r = union_stats(hits, which, kh, pool_ids, q_ids, gt, CAP_PER_SOURCE)
        r["combination"] = label
        out["rows"].append(r)
        print(f"  {label:<42} cand/S1={r['avg_candidates']:6.1f} "
              f"PR={r['pair_recall']:.4f} FCR={r['full_cluster_recall']:.4f} "
              f"ceiling={r['f05_ceiling']:.4f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    path = config.REPORT_DIR / "probe_blocking.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {path} in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
