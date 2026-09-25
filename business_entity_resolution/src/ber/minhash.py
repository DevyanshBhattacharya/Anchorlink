"""A fifth retriever: banded MinHash, i.e. locality-sensitive hashing.

**Why another retriever at all.**  The four sparse views already cover name,
address, both together and the consonant skeleton, and their union reaches pair
recall 0.928 on the full India training partition.  The 7% they miss is not
mostly a similarity problem — it is a *ranking* problem.  All four are top-k
retrievers, and in a pool where 35-44% of names repeat, a true match can sit at
rank 200 behind two hundred look-alikes and never be seen however good its
similarity is.  Adding k makes that better slowly and costs linearly.

LSH is not a ranked retriever.  A pair either collides in a band or it does not,
and whether it collides depends only on the two records, never on how much
company they have.  That is a different failure mode from the other four, which
is the whole reason to add it — the same argument the four-view union already
rests on.

It also sidesteps the query pruning the cosine views need at this scale: those
keep only a query's rarest terms within a budget on postings visited, measured to
cost 0.017 of pair recall on the name+address view alone.  A MinHash signature is
computed from the whole record.

**The construction.**  ``n_perm`` hash permutations over the record's character
n-gram set; the minimum under each permutation is one signature slot, and the
probability that two records agree on a slot is exactly their Jaccard similarity.
The signature is cut into ``bands`` groups of ``rows``, each group hashed to one
bucket key, and two records are candidates if they share **any** bucket.  So

    P(candidate) = 1 - (1 - J**rows) ** bands

which is a step around ``J* = (1/bands) ** (1/rows)``.  At the default 20 bands
of 3 that is J* = 0.37 — deliberately low, because this retriever exists to catch
the pairs the precise views already lost.

Bucket tables are :class:`ber.keys.KeyBlocks`, which is exactly the capped,
sorted hash-to-rows structure this needs and is already tested.
"""
from __future__ import annotations

import pickle
import time
import zlib
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .blocking import char_wb_ngrams
from .keys import KeyBlocks

#: Mersenne prime.  Universal hashing needs a prime modulus, and 2**31-1 keeps
#: (a*x + b) inside uint64 for 32-bit x without an overflow check.
_P = np.uint64((1 << 31) - 1)

#: 64-bit mixing constants (SplitMix64 / xxHash), for folding a band's rows into
#: one bucket key.
_MIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9),
        np.uint64(0x94D049BB133111EB), np.uint64(0xD6E8FEB86659FD93),
        np.uint64(0xA24BAED4963EE407), np.uint64(0x9FB21C651E98DF25))


@dataclass(frozen=True)
class MinHashConfig:
    """How one banded-MinHash index is built and queried."""
    name: str = "lsh"
    ngram: int = 4
    view: str = "index"          # same view vocabulary as IndexConfig
    bands: int = 20
    rows: int = 3
    #: rows kept per bucket.  A bucket holding thousands of records is a hash
    #: collision or a generic string, not evidence.
    max_bucket: int = 40
    seed: int = 20260925

    @property
    def n_perm(self) -> int:
        return self.bands * self.rows

    @property
    def threshold(self) -> float:
        """The Jaccard similarity at which a pair becomes likely to collide."""
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def analyse(self, text: str) -> List[str]:
        return char_wb_ngrams(text, self.ngram)


def permutations(cfg: MinHashConfig) -> Tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` for the universal family ``h_i(x) = (a_i x + b_i) mod P``."""
    rng = np.random.default_rng(cfg.seed)
    a = rng.integers(1, int(_P), size=cfg.n_perm, dtype=np.uint64)
    b = rng.integers(0, int(_P), size=cfg.n_perm, dtype=np.uint64)
    return a, b


class MinHashIndex:
    """Banded MinHash over one country partition of one source."""

    def __init__(self, cfg: MinHashConfig):
        self.cfg = cfg
        self.a, self.b = permutations(cfg)
        self.bands: List[KeyBlocks] = []
        self.n_docs = 0
        self._vocab: Dict[str, int] = {}

    # -- tokenising --------------------------------------------------------
    def _gram_ids(self, texts: Sequence[str], grow: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Flat gram-hash array plus per-record offsets.

        Grams are hashed to 31 bits rather than looked up in a vocabulary, so a
        query gram the pool never contained still participates — which is the
        point of a signature over the whole record.  ``zlib.crc32`` is used for
        the same reason it is used for the key blocks: it is stable across runs
        and machines, unlike Python's salted ``hash``.  The cache exists because
        the same few hundred thousand grams recur across millions of records.
        """
        cache = self._vocab
        ids = array("l")
        indptr = array("q", [0])
        for text in texts:
            for g in self.cfg.analyse(text):
                h = cache.get(g)
                if h is None:
                    h = zlib.crc32(g.encode("utf-8")) & 0x7FFFFFFF
                    if grow:
                        cache[g] = h
                ids.append(h)
            indptr.append(len(ids))
        return (np.frombuffer(ids, dtype=np.int64).astype(np.uint64),
                np.frombuffer(indptr, dtype=np.int64))

    def signatures(self, texts: Sequence[str], chunk: int = 50_000,
                   grow: bool = True) -> np.ndarray:
        """``[n, n_perm]`` uint32 signatures.

        One permutation at a time over the whole chunk's grams, not all
        permutations at once: the latter materialises ``nnz x n_perm`` 64-bit
        values, which is a gigabyte per 20k records.  Same arithmetic, 60x less
        of it resident.
        """
        cfg = self.cfg
        out = np.empty((len(texts), cfg.n_perm), dtype=np.uint32)
        for lo in range(0, len(texts), chunk):
            hi = min(lo + chunk, len(texts))
            g, indptr = self._gram_ids(texts[lo:hi], grow)
            n = hi - lo
            empty = np.diff(indptr) == 0
            if len(g) == 0:
                out[lo:hi] = 0
                continue
            # reduceat indexes `g` directly, so an empty trailing record whose
            # offset equals nnz would raise; clip and zero those rows after.
            starts = np.minimum(indptr[:-1], len(g) - 1).astype(np.intp)
            for i in range(cfg.n_perm):
                h = (self.a[i] * g + self.b[i]) % _P
                out[lo:hi, i] = np.minimum.reduceat(h, starts).astype(np.uint32)
            if empty.any():
                out[lo + np.flatnonzero(empty)] = 0
        return out

    def band_keys(self, sig: np.ndarray) -> np.ndarray:
        """``[n, bands]`` int64 bucket keys, one per band."""
        cfg = self.cfg
        keys = np.empty((sig.shape[0], cfg.bands), dtype=np.int64)
        for band in range(cfg.bands):
            # the band index salts the key, so band 3's buckets cannot be
            # confused with band 7's.  Masked in Python ints: numpy's uint64
            # scalar multiply warns on the wrap this relies on.
            salt = np.uint64(((band + 1) * int(_MIX[0])) & 0xFFFFFFFFFFFFFFFF)
            acc = np.full(sig.shape[0], salt, dtype=np.uint64)
            for r in range(cfg.rows):
                col = sig[:, band * cfg.rows + r].astype(np.uint64)
                acc ^= (col + np.uint64(1)) * _MIX[(r + 1) % len(_MIX)]
                acc ^= acc >> np.uint64(29)
            keys[:, band] = acc.astype(np.int64)
        return keys

    # -- build -------------------------------------------------------------
    def build(self, texts: Sequence[str], chunk: int = 50_000,
              verbose: bool = False) -> "MinHashIndex":
        cfg = self.cfg
        t0 = time.time()
        self.n_docs = len(texts)
        key_parts: List[np.ndarray] = []
        for lo in range(0, len(texts), chunk):
            hi = min(lo + chunk, len(texts))
            key_parts.append(self.band_keys(self.signatures(texts[lo:hi], chunk)))
            if verbose and (hi // chunk) % 10 == 0:
                print(f"    [{cfg.name}] signed {hi:,}/{len(texts):,} "
                      f"({hi / max(time.time() - t0, 1e-9):,.0f}/s)", flush=True)
        keys = np.vstack(key_parts) if key_parts else np.zeros((0, cfg.bands), np.int64)
        del key_parts
        rows = np.arange(self.n_docs, dtype=np.int32)
        self.bands = [KeyBlocks.from_arrays(keys[:, b], rows, cfg.max_bucket)
                      for b in range(cfg.bands)]
        del keys
        if verbose:
            nnz = sum(len(kb.rows) for kb in self.bands)
            print(f"    [{cfg.name}] {cfg.bands}x{cfg.rows} bands "
                  f"(J*={cfg.threshold:.2f}), {nnz:,} bucket rows, "
                  f"{self.nbytes() / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
        return self

    def nbytes(self) -> int:
        return int(sum(kb.uniq.nbytes + kb.rows.nbytes + kb.starts.nbytes
                       for kb in self.bands))

    # -- query -------------------------------------------------------------
    def query(self, texts: Sequence[str], k: int, chunk: int = 20_000,
              **_: object) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k colliding documents per query, ranked by how many bands agreed.

        Signature matches ``SparseIndex.query`` so the retrieval driver does not
        care which kind of index it is holding.  The score is the share of bands
        that collided, which is monotone in Jaccard and is what the model sees.
        """
        nq = len(texts)
        out_idx = np.full((nq, k), -1, dtype=np.int32)
        out_scr = np.zeros((nq, k), dtype=np.float32)
        if not self.bands:
            return out_idx, out_scr
        for lo in range(0, nq, chunk):
            hi = min(lo + chunk, nq)
            qkeys = self.band_keys(self.signatures(texts[lo:hi], chunk, grow=False))
            q_parts, c_parts = [], []
            for b, kb in enumerate(self.bands):
                q, c = _gather_bucket(kb, qkeys[:, b])
                if len(q):
                    q_parts.append(q)
                    c_parts.append(c)
            if not q_parts:
                continue
            q = np.concatenate(q_parts)
            c = np.concatenate(c_parts)
            del q_parts, c_parts
            # collisions per (query, candidate), then top-k within each query
            n_pool = self.n_docs
            uniq, counts = np.unique(q.astype(np.int64) * n_pool + c.astype(np.int64),
                                     return_counts=True)
            qq = (uniq // n_pool).astype(np.int64)
            cc = (uniq % n_pool).astype(np.int32)
            score = counts.astype(np.float32) / self.cfg.bands
            order = np.lexsort((-score, qq))
            qs, cs, ss = qq[order], cc[order], score[order]
            starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
            ends = np.r_[starts[1:], len(qs)]
            for s, e in zip(starts, ends):
                m = min(k, e - s)
                row = lo + int(qs[s])
                out_idx[row, :m] = cs[s:s + m]
                out_scr[row, :m] = ss[s:s + m]
        return out_idx, out_scr

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"cfg": asdict(self.cfg), "n_docs": self.n_docs,
                         "bands": [(kb.uniq, kb.rows, kb.starts) for kb in self.bands]},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "MinHashIndex":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls(MinHashConfig(**d["cfg"]))
        obj.n_docs = d["n_docs"]
        obj.bands = [KeyBlocks(u, r, s) for u, r, s in d["bands"]]
        return obj


def _gather_bucket(kb: KeyBlocks, qkeys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised many-query bucket lookup -> aligned (query index, row) arrays.

    ``KeyBlocks.lookup`` handles one query and loops in Python; twenty bands over
    twenty thousand queries is four hundred thousand of those, so the ragged
    gather is done with arithmetic instead.
    """
    if len(kb.uniq) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    pos = np.searchsorted(kb.uniq, qkeys)
    pos = np.clip(pos, 0, len(kb.uniq) - 1)
    ok = kb.uniq[pos] == qkeys
    if not ok.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    qi = np.flatnonzero(ok)
    p = pos[qi]
    sizes = (kb.starts[p + 1] - kb.starts[p]).astype(np.int64)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    q_out = np.repeat(qi, sizes)
    # offset of each gathered row inside its own bucket
    first = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=first[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(first[:-1], sizes)
    return q_out, kb.rows[np.repeat(kb.starts[p], sizes) + within]
