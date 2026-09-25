"""Deterministic key blocks — capped, and only ever a *source of extra recall*.

Keys add candidates for records whose text is corrupted but whose number and
name root survive ("Apex Digital" -> "Apex Didt1" at the same house number).
They are useless on their own — the best single exact key found 62% of true pairs
with 44 false candidates per entity — so each key block is capped and the hit is
also handed to the model as a feature.

Keys are stored as 64-bit hashes in sorted numpy arrays rather than a Python
dict: 2.3M keys cost 18 MB this way and ~300 MB as a dict.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .normalize import TokenRoles, house_numbers, skeleton, tokens

MAX_BLOCK = 50          # records per key; larger blocks carry no signal


def hash64(s: str) -> int:
    """Two salted CRC32s packed into a signed 64-bit int (numpy-safe)."""
    b = s.encode("utf-8")
    v = (zlib.crc32(b) << 32) | zlib.crc32(b + b"\x00salt")
    return v - (1 << 64) if v >= (1 << 63) else v


def record_keys(name: str, addr: str, country: str, roles: TokenRoles) -> List[int]:
    """The key set of one record: number+root, and the sorted core skeleton."""
    core = roles.core_tokens(name, country)
    out: List[int] = []
    if core:
        root = skeleton(core[0]) or core[0]
        for n in house_numbers(addr):
            out.append(hash64(f"n:{n}|{root}"))
        sk = " ".join(sorted(p for p in skeleton(" ".join(core)).split() if p))
        if sk:
            out.append(hash64(f"s:{sk}"))
    return out


class KeyBlocks:
    """key hash -> pool row ids, as a sorted (key, row) pair of arrays."""

    def __init__(self, uniq: np.ndarray, rows: np.ndarray, starts: np.ndarray):
        self.uniq = uniq            # [K] sorted unique key hashes
        self.starts = starts        # [K+1] offsets into rows
        self.rows = rows            # [nnz] pool row ids

    @classmethod
    def build(cls, names: Sequence[str], addrs: Sequence[str], country: str,
              roles: TokenRoles, max_block: int = MAX_BLOCK) -> "KeyBlocks":
        key_list: List[int] = []
        row_list: List[int] = []
        for i, (n, a) in enumerate(zip(names, addrs)):
            for k in record_keys(n, a, country, roles):
                key_list.append(k)
                row_list.append(i)
        if not key_list:
            empty_i = np.zeros(0, dtype=np.int64)
            return cls(empty_i, np.zeros(0, dtype=np.int32), np.zeros(1, dtype=np.int64))
        keys = np.asarray(key_list, dtype=np.int64)
        rows = np.asarray(row_list, dtype=np.int32)
        del key_list, row_list
        order = np.argsort(keys, kind="stable")
        keys, rows = keys[order], rows[order]
        uniq, starts_idx, counts = np.unique(keys, return_index=True, return_counts=True)
        # cap each block: keep the first `max_block` rows of every key
        keep = np.ones(len(keys), dtype=bool)
        big = np.flatnonzero(counts > max_block)
        for b in big:
            s = starts_idx[b]
            keep[s + max_block: s + counts[b]] = False
        rows = rows[keep]
        counts = np.minimum(counts, max_block)
        starts = np.zeros(len(uniq) + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        return cls(uniq, rows, starts)

    def lookup(self, key_hashes: Sequence[int]) -> np.ndarray:
        """Pool row ids hit by any of these keys (may repeat)."""
        if len(self.uniq) == 0 or not len(key_hashes):
            return np.zeros(0, dtype=np.int32)
        kh = np.asarray(key_hashes, dtype=np.int64)
        pos = np.searchsorted(self.uniq, kh)
        pos = np.clip(pos, 0, len(self.uniq) - 1)
        ok = self.uniq[pos] == kh
        pos = pos[ok]
        if not len(pos):
            return np.zeros(0, dtype=np.int32)
        return np.concatenate([self.rows[self.starts[p]:self.starts[p + 1]] for p in pos])
