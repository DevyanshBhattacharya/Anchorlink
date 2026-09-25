"""Per-split corpus statistics (Module 2, layer 2).

:class:`~ber.normalize.TokenRoles` is fitted over the whole pool of a split —
S1, S2 and S3 together, one country at a time — because that is what the
retrieval index sees.  These are unlabeled corpus statistics (IDF, edge-position
affixness, name frequency), the same thing a BM25 index computes, so fitting the
*test* statistics on test records is allowed and is what puts France on the same
scale as the US and India.  No label ever enters here.

Cached as a pickle under ``work/cache``; rebuilt with ``force=True``.
"""
from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

from . import config
from .data import build_cache, cache_path
from .normalize import TokenRoles, clear_caches

CHUNK = 400_000


def roles_path(split: str) -> Path:
    return config.CACHE_DIR / f"{split}_token_roles.pkl"


def build_roles(split: str, force: bool = False, verbose: bool = True) -> TokenRoles:
    """Fit and cache TokenRoles for one split, streaming every source file."""
    out = roles_path(split)
    if out.exists() and not force:
        with open(out, "rb") as fh:
            return pickle.load(fh)

    roles = TokenRoles()
    t0 = time.time()
    for source in ("S1", "S2", "S3"):
        build_cache(split, source)
        pf = pq.ParquetFile(cache_path(split, source))
        n = 0
        for batch in pf.iter_batches(batch_size=CHUNK, columns=["name", "addr", "country"]):
            d = batch.to_pydict()
            roles.partial_fit(d["name"], [str(c) for c in d["country"]], d["addr"])
            n += len(d["name"])
        if verbose:
            print(f"  [roles] {split}/{source}: {n:,} records, {time.time() - t0:.0f}s")
        del pf
        gc.collect()
    roles.finalize(min_df=2, min_addr_df=2)
    clear_caches()
    with open(out, "wb") as fh:
        pickle.dump(roles, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  [roles] {split}: finalised in {time.time() - t0:.0f}s -> {out.name} "
              f"({out.stat().st_size / 1e6:.0f} MB)")
    return roles


def describe_affixes(roles: TokenRoles, top: int = 20) -> Dict[str, List[str]]:
    """The legal forms the edge statistic found, most frequent first, per country.

    This is the evidence that nothing language-specific is hard-coded: the same
    statistic recovers llc/inc/corp, limited/ltd/llp and sarl/sas/eurl/sasu/sci.
    """
    out: Dict[str, List[str]] = {}
    for c, toks in roles.affix.items():
        idf = roles.idf.get(c, {})
        nd = roles.n_docs.get(c, 1)
        scored = sorted(toks, key=lambda w: idf.get(w, 99.0))
        out[c] = [f"{w} {np.exp(-idf[w]) * 100:.1f}%" if w in idf else w
                  for w in scored[:top]]
    return out
