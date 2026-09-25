"""Module 3 — sparse retrieval, one country partition at a time.

Ranked retrieval on **name + address**, never name alone: 35–44% of S1 names are
shared by another business, and name-only retrieval found 62% of Kerala pairs at
top-20 where name+address found 97.6%.

Two things make the doc's recipe survive the jump from a 269k-record state slice
to a 2.4M-record country partition:

* **An absolute document-frequency cap** instead of the relative ``max_df=2%``.
  2% of a state slice is ~5k documents; 2% of the India test pool is 46k, which
  would leave posting lists two orders of magnitude longer than the ones the
  recipe was tuned on.
* **Rare-term query pruning.**  A query keeps only its rarest terms, up to a
  budget on the number of postings visited.  Cosine is dominated by rare terms
  anyway, and the query's L2 norm is a per-query constant that cannot reorder its
  own top-k, so the ranking is close to the unpruned one at a fraction of the
  cost.

Everything is per country partition, keyed on the exact country string, so a
label that never appeared in training (France) simply gets its own index.
"""
from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from array import array
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from . import config
from .normalize import clean_text, skeleton

#: Threads for the sparse top-k product.  sparse_dot_topn defaults to one, which
#: leaves nine of this machine's ten cores idle during the dominant cost of
#: blocking.
N_THREADS = max(1, (os.cpu_count() or 4))

_OMP_WARNED = False


def check_openmp_health() -> str | None:
    """Warn once if another OpenMP runtime is already loaded in this process.

    LightGBM (and scikit-learn) ship their own libomp.  With two copies loaded,
    the sparse top-k product silently drops to a single thread and blocking runs
    about ten times slower — measured, not theoretical (PROGRESS.md §7).  Import
    order is the fix; this is the alarm that says the fix has been undone.
    """
    global _OMP_WARNED
    if _OMP_WARNED:
        return None
    offenders = [m for m in ("lightgbm", "sklearn") if m in sys.modules]
    _OMP_WARNED = True
    if offenders:
        msg = (f"WARNING: {', '.join(offenders)} already imported; each ships its own "
               f"OpenMP runtime and the sparse top-k product may fall back to a "
               f"single thread. Run retrieval before importing them.")
        print(msg, flush=True)
        return msg
    return None


# --------------------------------------------------------------- analysers
def char_wb_ngrams(text: str, n: int = 3) -> List[str]:
    """Character n-grams inside word boundaries, as scikit-learn's ``char_wb``.

    Word-bounded grams beat plain character grams here because business fields
    are reordered constantly ("OH, Columbus, 5559 Kingsmere Avenue"), and a gram
    that straddles two words encodes an order that does not survive the noise.
    """
    out: List[str] = []
    for w in text.split():
        s = " " + w + " "
        m = len(s)
        if m <= n:
            out.append(s)
        else:
            out.extend(s[i:i + n] for i in range(m - n + 1))
    return out


def word_ngrams(text: str, n: int = 1) -> List[str]:
    ws = text.split()
    if n == 1 or len(ws) <= n:
        return ws if n == 1 else ([" ".join(ws)] if ws else [])
    return [" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)]


@dataclass(frozen=True)
class IndexConfig:
    """How one sparse index is built and queried."""
    name: str = "char3"
    kind: str = "char"                 # "char" | "word" | "char+word"
    ngram: int = 3
    view: str = "index"                # "index" (folded) | "skeleton"
    min_df: int = 2
    max_df_abs: int = 20_000           # absolute posting-list cap
    max_df_frac: float = 0.02          # and never more than this share of the pool
    sublinear_tf: bool = True
    query_budget: int = 40_000         # postings a single query may visit
    min_query_terms: int = 6
    max_query_terms: int = 64

    def analyse(self, text: str) -> List[str]:
        if self.kind == "char":
            return char_wb_ngrams(text, self.ngram)
        if self.kind == "word":
            return word_ngrams(text, self.ngram)
        return word_ngrams(text, 1) + char_wb_ngrams(text, self.ngram)


def make_view(cfg: IndexConfig, name: str, addr: str) -> str:
    """The text one index sees.

    ``name``/``addr`` alone matter because the two fields fail independently: a
    record with a transliterated name but an intact address is found by the
    address view, and one with a corrupted address but an exact name by the name
    view.  ``skeleton`` is the cross-script view; ``skelname`` is the same on the
    name alone.
    """
    v = cfg.view
    if v == "index":
        return f"{clean_text(name)} {clean_text(addr)}".strip()
    if v == "skeleton":
        return f"{skeleton(name)} {skeleton(addr)}".strip()
    if v == "name":
        return clean_text(name)
    if v == "addr":
        return clean_text(addr)
    if v == "skelname":
        return skeleton(name)
    raise ValueError(f"unknown view {v!r}")


# --------------------------------------------------------------- the index
class SparseIndex:
    """A TF-IDF index over one country partition of one source."""

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None       # [V] float32
        self.df: np.ndarray | None = None        # [V] int32
        self.Dt: sp.csr_matrix | None = None     # [V, ndocs] for A @ Dt
        self.n_docs = 0

    # -- build ------------------------------------------------------------
    def build(self, texts: Sequence[str], verbose: bool = False) -> "SparseIndex":
        cfg = self.cfg
        t0 = time.time()
        vocab: Dict[str, int] = {}
        cols = array("i")
        vals = array("i")
        indptr = array("q", [0])
        for text in texts:
            grams = cfg.analyse(text)
            if grams:
                counts: Dict[int, int] = {}
                for g in grams:
                    gid = vocab.get(g)
                    if gid is None:
                        gid = len(vocab)
                        vocab[g] = gid
                    counts[gid] = counts.get(gid, 0) + 1
                cols.extend(counts.keys())
                vals.extend(counts.values())
            indptr.append(len(cols))
        self.n_docs = len(indptr) - 1
        V = len(vocab)
        if verbose:
            print(f"    [{cfg.name}] tokenised {self.n_docs:,} docs, V={V:,}, "
                  f"nnz={len(cols):,} in {time.time() - t0:.0f}s")

        data = np.frombuffer(vals, dtype=np.int32)
        indices = np.frombuffer(cols, dtype=np.int32)
        indptr_np = np.frombuffer(indptr, dtype=np.int64)
        D = sp.csr_matrix((data.astype(np.float32), indices, indptr_np),
                          shape=(self.n_docs, V))
        del data, indices, cols, vals
        gc.collect()

        df = np.bincount(D.indices, minlength=V).astype(np.int64)
        cap = min(cfg.max_df_abs, max(1, int(cfg.max_df_frac * self.n_docs)))
        keep = (df >= cfg.min_df) & (df <= cap)
        if not keep.any():
            keep = df >= 1
        remap = np.full(V, -1, dtype=np.int32)
        new_ids = np.flatnonzero(keep)
        remap[new_ids] = np.arange(len(new_ids), dtype=np.int32)

        D = _select_columns(D, keep, remap, len(new_ids))
        self.vocab = {g: int(remap[i]) for g, i in vocab.items() if keep[i]}
        self.df = df[new_ids].astype(np.int32)
        self.idf = (np.log(self.n_docs / np.maximum(self.df, 1)) + 1.0).astype(np.float32)

        if cfg.sublinear_tf:
            np.log(D.data, out=D.data)
            D.data += 1.0
        D.data *= self.idf[D.indices]
        _l2_normalise_rows(D)
        self.Dt = D.T.tocsr()
        del D
        gc.collect()
        if verbose:
            print(f"    [{cfg.name}] index ready: V={len(new_ids):,} "
                  f"nnz={self.Dt.nnz:,} cap={cap:,} in {time.time() - t0:.0f}s "
                  f"({self.nbytes() / 1e9:.2f} GB)")
        return self

    def nbytes(self) -> int:
        if self.Dt is None:
            return 0
        return int(self.Dt.data.nbytes + self.Dt.indices.nbytes + self.Dt.indptr.nbytes)

    # -- query ------------------------------------------------------------
    def transform(self, texts: Sequence[str], prune: bool = True) -> sp.csr_matrix:
        """Vectorise queries, keeping only the rarest terms within the budget."""
        cfg = self.cfg
        V = len(self.idf)
        cols = array("i")
        vals = array("f")
        indptr = array("q", [0])
        df = self.df
        budget, min_terms, max_terms = cfg.query_budget, cfg.min_query_terms, cfg.max_query_terms
        for text in texts:
            counts: Dict[int, int] = {}
            for g in cfg.analyse(text):
                gid = self.vocab.get(g)
                if gid is not None:
                    counts[gid] = counts.get(gid, 0) + 1
            if counts:
                ids = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
                tf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
                if prune and len(ids) > min_terms:
                    order = np.argsort(df[ids], kind="stable")
                    ids, tf = ids[order], tf[order]
                    cum = np.cumsum(df[ids].astype(np.int64))
                    n_keep = int(np.searchsorted(cum, budget, side="right"))
                    n_keep = max(min_terms, min(n_keep, max_terms, len(ids)))
                    ids, tf = ids[:n_keep], tf[:n_keep]
                elif len(ids) > max_terms:
                    order = np.argsort(df[ids], kind="stable")[:max_terms]
                    ids, tf = ids[order], tf[order]
                if cfg.sublinear_tf:
                    tf = np.log(tf) + 1.0
                w = tf * self.idf[ids]
                n = np.sqrt(float(w @ w))
                if n > 0:
                    w /= n
                cols.extend(ids.tolist())
                vals.extend(w.tolist())
            indptr.append(len(cols))
        return sp.csr_matrix(
            (np.frombuffer(vals, dtype=np.float32),
             np.frombuffer(cols, dtype=np.int32),
             np.frombuffer(indptr, dtype=np.int64)),
            shape=(len(indptr) - 1, V))

    def query(self, texts: Sequence[str], k: int, chunk: int = 20_000,
              threshold: float = 0.0, n_threads: int | None = None,
              prune: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Top-k documents per query.

        Returns ``(idx, score)`` arrays of shape ``[nq, k]``; unfilled slots are
        ``-1`` / ``0.0``.
        """
        check_openmp_health()
        nq = len(texts)
        out_idx = np.full((nq, k), -1, dtype=np.int32)
        out_scr = np.zeros((nq, k), dtype=np.float32)
        for lo in range(0, nq, chunk):
            hi = min(lo + chunk, nq)
            Q = self.transform(texts[lo:hi], prune=prune)
            S = sp_matmul_topn(Q, self.Dt, top_n=k, threshold=threshold or None,
                               sort=True, n_threads=n_threads or N_THREADS)
            _fill_topk(S, out_idx, out_scr, lo)
            del Q, S
        return out_idx, out_scr

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "cfg": asdict(self.cfg), "vocab": self.vocab,
                "idf": self.idf, "df": self.df, "n_docs": self.n_docs,
                "data": self.Dt.data, "indices": self.Dt.indices,
                "indptr": self.Dt.indptr, "shape": self.Dt.shape,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls(IndexConfig(**d["cfg"]))
        obj.vocab, obj.idf, obj.df, obj.n_docs = d["vocab"], d["idf"], d["df"], d["n_docs"]
        obj.Dt = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
        return obj


# --------------------------------------------------------------- helpers
def _row_reduce(values: np.ndarray, indptr: np.ndarray, n_rows: int) -> np.ndarray:
    """Per-row sums of a CSR value array, safe for empty rows.

    ``np.add.reduceat`` indexes ``values`` directly, so an empty row whose
    offset equals ``nnz`` — which is every trailing empty row — raises
    IndexError.  Clipping the offsets and zeroing the empty rows afterwards is
    exact, because a clipped offset only ever belongs to a row we then zero.
    """
    counts = np.diff(indptr)
    if len(values) == 0:
        return np.zeros(n_rows, dtype=np.float64)
    starts = np.minimum(indptr[:-1], len(values) - 1)
    out = np.add.reduceat(values, starts.astype(np.intp))
    out[counts == 0] = 0
    return out


def _select_columns(D: sp.csr_matrix, keep: np.ndarray, remap: np.ndarray,
                    n_new: int) -> sp.csr_matrix:
    """Drop columns from a CSR matrix without materialising a dense mask copy."""
    mask = keep[D.indices]
    new_indices = remap[D.indices[mask]]
    new_data = D.data[mask]
    kept_per_row = _row_reduce(mask.astype(np.int64), D.indptr, D.shape[0])
    new_indptr = np.zeros(D.shape[0] + 1, dtype=np.int64)
    np.cumsum(kept_per_row.astype(np.int64), out=new_indptr[1:])
    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=(D.shape[0], n_new))


def _l2_normalise_rows(D: sp.csr_matrix) -> None:
    if D.nnz == 0:
        return
    norms = np.sqrt(_row_reduce(D.data * D.data, D.indptr, D.shape[0]))
    counts = np.diff(D.indptr)
    norms[counts == 0] = 1.0
    norms[norms == 0] = 1.0
    D.data /= np.repeat(norms, counts).astype(D.data.dtype)


def _fill_topk(S: sp.csr_matrix, out_idx: np.ndarray, out_scr: np.ndarray,
               offset: int) -> None:
    k = out_idx.shape[1]
    indptr, indices, data = S.indptr, S.indices, S.data
    for r in range(S.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        m = min(k, b - a)
        if m:
            out_idx[offset + r, :m] = indices[a:a + m]
            out_scr[offset + r, :m] = data[a:a + m]
