"""Scoring a whole partition without ever holding its feature matrix in memory.

Three passes over the feature shards of one country:

1. **M1** on each shard, keeping only the probability (4 bytes per pair).
2. **meta** over the whole country at once — rank, gap, exclusivity and the
   competitor counts need the full table, but only need the probabilities.
3. **M2** on each shard again, with its slice of the meta block attached, then
   calibration and the exclusivity correction.

Shards are written by :mod:`ber.featurize` in S1 order, so concatenating them
gives a table already grouped by S1, which is what the meta and decision layers
assume.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .decide import exclusivity_arrays
from .featurize import feature_dir
from .features import build_meta_features
from .pipeline import STAGE2_FEATURES
from .train import Stack


def _shards(d: Path) -> List[Tuple[Path, Path]]:
    xs = sorted(d.glob("X-*.npy"))
    ks = sorted(d.glob("K-*.parquet"))
    if len(xs) != len(ks):
        raise RuntimeError(f"{d}: {len(xs)} feature shards vs {len(ks)} key shards")
    return list(zip(xs, ks))


def score_country(split: str, country: str, bcfg, stack: Stack,
                  verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(keys, p_calibrated, p_exclusive)`` for one country partition."""
    d = feature_dir(split, country, bcfg)
    shards = _shards(d)
    if not shards:
        raise FileNotFoundError(f"no feature shards in {d}")
    t0 = time.time()

    keys: List[pd.DataFrame] = []
    p1_parts: List[np.ndarray] = []
    for xp, kp in shards:
        X = np.load(xp, mmap_mode="r")
        p1_parts.append(stack.m1.raw_predict(np.asarray(X)).astype(np.float32))
        keys.append(pd.read_parquet(kp))
        del X
        gc.collect()
    K = pd.concat(keys, ignore_index=True)
    p1 = np.concatenate(p1_parts)
    del keys, p1_parts
    gc.collect()

    s1 = K["s1"].to_numpy()
    _, s1_codes = np.unique(s1, return_inverse=True)
    ckey = K["src"].to_numpy().astype(np.int64) * (1 << 32) + K["cid"].to_numpy().astype(np.int64)
    _, c_codes = np.unique(ckey, return_inverse=True)
    del ckey
    M = build_meta_features(s1_codes.astype(np.int64), c_codes.astype(np.int64),
                            K["src"].to_numpy().astype(np.int8), p1.astype(np.float64))
    del p1
    gc.collect()

    p2_parts: List[np.ndarray] = []
    off = 0
    for xp, _ in shards:
        X = np.load(xp)
        n = X.shape[0]
        p2_parts.append(stack.m2.raw_predict(np.hstack([X, M[off:off + n]])).astype(np.float32))
        off += n
        del X
        gc.collect()
    p2 = np.concatenate(p2_parts)
    del p2_parts, M
    gc.collect()

    p = (np.clip(stack.calibrator.predict(np.clip(p2, 0, 1)), 1e-6, 1 - 1e-6)
         if stack.calibrator is not None else np.clip(p2, 1e-6, 1 - 1e-6))
    pe = np.clip(exclusivity_arrays(s1_codes.astype(np.int64), c_codes.astype(np.int64), p),
                 1e-6, 1 - 1e-6)
    if verbose:
        print(f"[score] {split}/{country}: {len(K):,} pairs in {time.time() - t0:.0f}s",
              flush=True)
    return K, p, pe
