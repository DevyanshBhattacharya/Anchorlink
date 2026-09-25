"""Attach ground-truth labels to a candidate table, without a Python dict.

The 7,638,365 training pairs are packed into sorted int64 keys, so labelling a
30M-row candidate table is two numpy calls instead of 30M dict lookups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from . import config
from .data import load_gt_arrays

_SRC_BIT = {2: 0, 3: 1}


def pack(s1: np.ndarray, src: np.ndarray, cid: np.ndarray) -> np.ndarray:
    """(s1, src, cid) -> one int64.  s1 < 2**30 and cid < 2**32 in this data."""
    return (s1.astype(np.int64) << 33) | ((src.astype(np.int64) - 2) << 32) | cid.astype(np.int64)


class GroundTruth:
    def __init__(self, gt_path: Path | None = None):
        s1, mid, msrc = load_gt_arrays(gt_path)
        self.keys = np.sort(pack(s1, msrc, mid))
        self.s1_ids, self.sizes = np.unique(s1, return_counts=True)
        self._all_s1: np.ndarray | None = None

    def label(self, s1: np.ndarray, src: np.ndarray, cid: np.ndarray) -> np.ndarray:
        k = pack(np.asarray(s1), np.asarray(src), np.asarray(cid))
        pos = np.searchsorted(self.keys, k)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        return (self.keys[pos] == k).astype(np.int8)

    def cluster_size(self, s1: np.ndarray) -> np.ndarray:
        """Number of true matches per S1 id (0 for singletons)."""
        s1 = np.asarray(s1, dtype=np.int32)
        pos = np.searchsorted(self.s1_ids, s1)
        pos = np.clip(pos, 0, len(self.s1_ids) - 1)
        hit = self.s1_ids[pos] == s1
        out = np.zeros(len(s1), dtype=np.int32)
        out[hit] = self.sizes[pos[hit]]
        return out
