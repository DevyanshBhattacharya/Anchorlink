"""Turn candidate tables into feature matrices.

Work is done in blocks of S1 entities so peak memory stays flat: a block loads
only the record views it needs, computes its rows and frees them.  Blocks are
independent, so they parallelise over processes; the parent keeps the partition
text and ships each worker only the strings its block touches, which keeps
worker memory at a few hundred MB instead of a copy of the pool.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .data import Partition, load_partition
from .features import (FEATURE_VERSION, N_PAIR, I_RETRIEVAL, I_SET, PAIR_FEATURES,
                       RecordView, jaccard, pair_feature_row)
from .normalize import TokenRoles
from .retrieval import BlockingConfig, RETRIEVAL_COLUMNS, cand_dir

RRF_COL = RETRIEVAL_COLUMNS.index("rrf")
PRE_COL = RETRIEVAL_COLUMNS.index("pre")

_WORKER: Dict[str, object] = {}


# ------------------------------------------------------------------ block job
@dataclass
class Block:
    """One unit of feature work: a contiguous run of S1 groups."""
    country: str
    q_name: List[str]
    q_addr: List[str]
    c_name: List[str]
    c_addr: List[str]
    q_row: np.ndarray        # [n] index into q_name
    c_row: np.ndarray        # [n] index into c_name
    src: np.ndarray          # [n] int8, 2 or 3
    retr: np.ndarray         # [n, 7] float32 retrieval columns
    bounds: np.ndarray       # group offsets over the rows (sorted by S1)


def compute_block(block: Block, roles: TokenRoles) -> np.ndarray:
    n = len(block.q_row)
    X = np.empty((n, N_PAIR), dtype=np.float32)
    country = block.country
    qv = [RecordView(nm, ad, country, roles) for nm, ad in zip(block.q_name, block.q_addr)]
    cv = [RecordView(nm, ad, country, roles) for nm, ad in zip(block.c_name, block.c_addr)]
    row = np.empty(N_PAIR, dtype=np.float64)
    q_row, c_row = block.q_row, block.c_row
    for i in range(n):
        pair_feature_row(qv[q_row[i]], cv[c_row[i]], roles, country, row)
        X[i, :I_RETRIEVAL] = row[:I_RETRIEVAL]
    X[:, I_RETRIEVAL:I_SET] = block.retr

    # set-level columns: support against the top candidate of the other source,
    # plus the shape of this S1's candidate list
    rrf = block.retr[:, RRF_COL]
    # Which other-source candidate to take support from is a ranking question,
    # and the pre-ranker is a model trained to answer it while RRF is an
    # unweighted rank sum.  Use `pre` when it exists (it does on a pruned table,
    # which is what the matcher scores) and fall back to RRF otherwise, so the
    # raw table still featurises.
    pre = block.retr[:, PRE_COL]
    anchor = pre if np.isfinite(pre).any() else rrf
    src = block.src
    for a, b in zip(block.bounds[:-1], block.bounds[1:]):
        if b <= a:
            continue
        seg = slice(a, b)
        seg_rrf, seg_src = rrf[seg], src[seg]
        seg_anchor = anchor[seg]
        top = {}
        for s in (2, 3):
            m = seg_src == s
            if m.any():
                top[s] = int(np.flatnonzero(m)[np.argmax(seg_anchor[m])]) + a
        order = np.argsort(-seg_rrf, kind="stable")
        ranks = np.empty(b - a, dtype=np.float32)
        ranks[order] = np.arange(b - a, dtype=np.float32)
        best = seg_rrf.max() if b > a else 0.0
        for j in range(a, b):
            other = top.get(3 if src[j] == 2 else 2)
            if other is None or other == j:
                X[j, I_SET] = np.nan
                X[j, I_SET + 1] = np.nan
            else:
                sn, sa = (jaccard(cv[c_row[j]].name_grams, cv[c_row[other]].name_grams),
                          jaccard(cv[c_row[j]].addr_grams, cv[c_row[other]].addr_grams))
                X[j, I_SET] = sn
                X[j, I_SET + 1] = sa
        X[seg, I_SET + 2] = ranks
        X[seg, I_SET + 3] = best - seg_rrf
        X[seg, I_SET + 4] = float(b - a)
    return X


def _init_worker(roles_path: str, country: str) -> None:
    import pickle
    with open(roles_path, "rb") as fh:
        roles: TokenRoles = pickle.load(fh)
    # a worker only ever sees one partition; the rest of the tables are ballast
    _WORKER["roles"] = roles.subset(country)
    _WORKER["country"] = country
    del roles


def _run_block(block: Block) -> np.ndarray:
    return compute_block(block, _WORKER["roles"])


# ------------------------------------------------------------------ driver
def feature_dir(split: str, country: str, bcfg: BlockingConfig) -> Path:
    """Feature shards, keyed by blocking config **and** feature-set version.

    Adding or reordering a column changes what a shard means; without the
    version in the path an old shard would be stacked with a new one and the
    model would be reading different features in different rows.
    """
    safe = country.replace("/", "_").replace(" ", "_")
    return (config.WORK_DIR / "features" / f"{bcfg.fingerprint()}_f{FEATURE_VERSION}"
            / f"{split}_{safe}")


def _lookup(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(ids, kind="stable")
    return ids[order], order


def _rows_for(sorted_ids: np.ndarray, order: np.ndarray, want: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_ids, want)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    bad = sorted_ids[pos] != want
    rows = order[pos]
    if bad.any():
        rows = rows.copy()
        rows[bad] = -1
    return rows


def _retr_matrix(rows: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Retrieval columns as float32; ``pre`` is absent until the pre-ranker ran."""
    M = np.empty((len(rows), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        M[:, j] = rows[c].to_numpy(dtype=np.float32) if c in rows.columns else np.nan
    return M


def iter_blocks(split: str, country: str, bcfg: BlockingConfig,
                block_s1: int = 4000, pruned: bool = True
                ) -> Iterable[Tuple[Block, pd.DataFrame]]:
    """Yield ``(block, rows)`` — the feature job and the candidate rows behind it."""
    d = cand_dir(split, country, bcfg, pruned=pruned)
    parts = sorted(d.glob("S*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no candidate shards in {d}")
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    df.sort_values(["s1", "src", "cid"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)

    s1p = load_partition(split, "S1", country)
    pools = {2: load_partition(split, "S2", country), 3: load_partition(split, "S3", country)}
    s1_sorted, s1_order = _lookup(s1p.ids)
    pool_lut = {s: _lookup(p.ids) for s, p in pools.items()}

    s1_arr = df["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1_arr[1:] != s1_arr[:-1]])
    starts = np.r_[starts, len(s1_arr)]
    n_groups = len(starts) - 1

    retr_cols = list(RETRIEVAL_COLUMNS)
    for g0 in range(0, n_groups, block_s1):
        g1 = min(g0 + block_s1, n_groups)
        lo, hi = int(starts[g0]), int(starts[g1])
        rows = df.iloc[lo:hi]
        uq, q_row_local = np.unique(rows["s1"].to_numpy(), return_inverse=True)
        q_rows = _rows_for(s1_sorted, s1_order, uq)
        src = rows["src"].to_numpy().astype(np.int8)
        cid = rows["cid"].to_numpy()
        # candidates are keyed by (src, cid) so the two sources cannot collide
        ckey = src.astype(np.int64) * (1 << 32) + cid.astype(np.int64)
        uck, c_row_local = np.unique(ckey, return_inverse=True)
        c_src = (uck >> 32).astype(np.int8)
        c_ids = (uck & 0xFFFFFFFF).astype(np.int32)
        # one searchsorted per source, not one per candidate: the per-candidate
        # version cost ~10 us each, which is minutes per country at this scale
        c_name: List[str] = [""] * len(c_ids)
        c_addr: List[str] = [""] * len(c_ids)
        for s in (2, 3):
            sel = np.flatnonzero(c_src == s)
            if not len(sel):
                continue
            ps, po = pool_lut[s]
            rows_s = _rows_for(ps, po, c_ids[sel])
            names_s, addrs_s = pools[s].names, pools[s].addrs
            for pos, r in zip(sel, rows_s):
                if r >= 0:
                    c_name[pos] = names_s[r]
                    c_addr[pos] = addrs_s[r]
        block = Block(
            country=country,
            q_name=[s1p.names[r] if r >= 0 else "" for r in q_rows],
            q_addr=[s1p.addrs[r] if r >= 0 else "" for r in q_rows],
            c_name=c_name, c_addr=c_addr,
            q_row=q_row_local.astype(np.int32), c_row=c_row_local.astype(np.int32),
            src=src,
            retr=_retr_matrix(rows, retr_cols),
            bounds=(starts[g0:g1 + 1] - lo).astype(np.int64),
        )
        yield block, rows


def featurize_partition(split: str, country: str, bcfg: BlockingConfig,
                        roles_path: Path, block_s1: int = 4000,
                        n_workers: int | None = None, verbose: bool = True,
                        force: bool = False, pruned: bool = True) -> Path:
    out = feature_dir(split, country, bcfg)
    out.mkdir(parents=True, exist_ok=True)
    src_done = cand_dir(split, country, bcfg, pruned=pruned) / "_DONE"
    stamp = src_done.read_text().strip() if src_done.exists() else "?"
    done = out / "_DONE"
    if done.exists() and not force:
        if done.read_text().strip() == stamp:
            return out
        for stale in list(out.glob("X-*.npy")) + list(out.glob("K-*.parquet")):
            stale.unlink()
        done.unlink()

    n_workers = n_workers if n_workers is not None else max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    keys: List[pd.DataFrame] = []
    mats: List[np.ndarray] = []
    n_rows = 0

    def flush(part: int) -> None:
        nonlocal keys, mats
        if not mats:
            return
        X = np.vstack(mats)
        K = pd.concat(keys, ignore_index=True)
        np.save(out / f"X-{part:04d}.npy", X)
        K.to_parquet(out / f"K-{part:04d}.parquet", index=False, compression="zstd")
        keys, mats = [], []

    part = 0
    if n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_workers, initializer=_init_worker,
                        initargs=(str(roles_path), country))
        try:
            pending = []
            for block, rows in iter_blocks(split, country, bcfg, block_s1, pruned):
                pending.append((pool.apply_async(_run_block, (block,)),
                                rows[["s1", "src", "cid"]].reset_index(drop=True)))
                while len(pending) >= n_workers * 2:
                    res, k = pending.pop(0)
                    mats.append(res.get()); keys.append(k); n_rows += len(k)
                    if n_rows and sum(m.shape[0] for m in mats) >= 4_000_000:
                        flush(part); part += 1
            for res, k in pending:
                mats.append(res.get()); keys.append(k); n_rows += len(k)
                if sum(m.shape[0] for m in mats) >= 4_000_000:
                    flush(part); part += 1
        finally:
            pool.close(); pool.join()
    else:
        import pickle
        with open(roles_path, "rb") as fh:
            roles = pickle.load(fh)
        for block, rows in iter_blocks(split, country, bcfg, block_s1, pruned):
            mats.append(compute_block(block, roles))
            keys.append(rows[["s1", "src", "cid"]].reset_index(drop=True))
            n_rows += len(rows)
            if sum(m.shape[0] for m in mats) >= 4_000_000:
                flush(part); part += 1
    flush(part)
    done.write_text(stamp + "\n")
    if verbose:
        print(f"[featurize] {split}/{country}: {n_rows:,} pairs in "
              f"{time.time() - t0:.0f}s ({n_workers} workers)", flush=True)
    return out


def load_features(split: str, country: str, bcfg: BlockingConfig,
                  s1_ids: np.ndarray | None = None
                  ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Stack the feature shards, optionally keeping only some S1 entities.

    Filtering happens **per shard**, before stacking: the full India test matrix
    is around 10M rows by 48 float32, and loading it whole only to throw most of
    it away doubles peak memory for no reason.
    """
    d = feature_dir(split, country, bcfg)
    xs = sorted(d.glob("X-*.npy"))
    ks = sorted(d.glob("K-*.parquet"))
    if not xs:
        raise FileNotFoundError(f"no feature shards in {d}")
    if len(xs) != len(ks):
        raise RuntimeError(f"{d}: {len(xs)} feature shards vs {len(ks)} key shards")
    want = None if s1_ids is None else np.asarray(s1_ids, dtype=np.int32)
    X_parts, K_parts = [], []
    for xp, kp in zip(xs, ks):
        Ki = pd.read_parquet(kp)
        if want is None:
            X_parts.append(np.load(xp))
            K_parts.append(Ki)
            continue
        m = np.isin(Ki["s1"].to_numpy(), want)
        if not m.any():
            continue
        Xi = np.load(xp, mmap_mode="r")
        X_parts.append(np.ascontiguousarray(Xi[m]))
        K_parts.append(Ki.loc[m].reset_index(drop=True))
        del Xi
    if not X_parts:
        raise ValueError(f"{d}: no rows matched the requested S1 ids")
    return np.vstack(X_parts), pd.concat(K_parts, ignore_index=True)
