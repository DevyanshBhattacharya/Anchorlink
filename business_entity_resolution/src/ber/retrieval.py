"""Candidate generation: four ranked retrievers per country partition, fused.

Measured on the full India training partition (4,133,346 S2+S3 records, 4,000
held-out queries searching the whole pool) — see RESULTS.md for the table:

| retriever (top-20 per source)   | pair recall | full-cluster recall |
| ------------------------------- | ----------- | ------------------- |
| name+address                    | 0.891       | 0.737               |
| address only                    | 0.818       | 0.579               |
| consonant skeleton              | 0.829       | 0.609               |
| name only                       | 0.558       | 0.295               |
| **union of the four**           | **0.947**   | **0.855**           |

The four fail independently, which is the whole point: a record with a
transliterated name but an intact address is found by the address view, one with
a corrupted address by the name view, and a Devanagari name by the skeleton.
Name-only retrieval is weak on its own (35–44% of names repeat) yet still adds
5 points of full-cluster recall to the union.

Ranks are fused with reciprocal-rank fusion (Cormack et al., SIGIR 2009, k=60),
capped deterministic key blocks are unioned in, and a learned **pre-ranker**
prunes the union to the list the matcher actually scores.  That pruned list is
what ``candidate_pairs.tsv`` records.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .blocking import IndexConfig, SparseIndex, make_view
from .data import Partition, load_partition
from .keys import MAX_BLOCK as DEFAULT_KEY_BLOCK_CAP, KeyBlocks, record_keys
from .normalize import TokenRoles

RRF_K = 60

#: The four retriever views.  ``tag`` is the column prefix in the candidate table.
#:
#: The document-frequency cap is **per view**, not global.  A word+char-4-gram
#: index over name+address has ~480k distinct terms, so capping posting lists at
#: 40k documents keeps every discriminative term.  The consonant-skeleton index
#: has only ~11k possible terms (consonants only, three at a time), so the same
#: absolute cap deletes almost the whole vocabulary and pair recall collapses
#: from 0.83 to 0.59.  The skeleton therefore caps by fraction of the pool.
RETRIEVERS: Tuple[Tuple[str, IndexConfig], ...] = (
    ("na", IndexConfig(name="na4", kind="char+word", ngram=4, view="index",
                       max_df_abs=20_000, max_df_frac=0.02)),
    ("nm", IndexConfig(name="nm4", kind="char+word", ngram=4, view="name",
                       max_df_abs=20_000, max_df_frac=0.02)),
    ("ad", IndexConfig(name="ad4", kind="char+word", ngram=4, view="addr",
                       max_df_abs=20_000, max_df_frac=0.02)),
    ("sk", IndexConfig(name="sk3", kind="char", ngram=3, view="skeleton",
                       max_df_abs=200_000, max_df_frac=0.03)),
)
RETRIEVER_TAGS = tuple(t for t, _ in RETRIEVERS)

#: Numeric columns of the candidate table, in the order the feature matrix wants.
RETRIEVAL_COLUMNS: Tuple[str, ...] = tuple(
    c for tag in RETRIEVER_TAGS for c in (f"{tag}_score", f"{tag}_rank")
) + ("key_hit", "rrf", "nret", "pre")


@dataclass
class BlockingConfig:
    """Everything that decides what lands in ``candidate_pairs.tsv``."""
    k_per_retriever: int = 30
    k_keys: int = 20
    use_keys: bool = True
    max_union_per_source: int = 80
    #: multiplies each retriever's own df cap, so one knob still sweeps them all
    df_scale: float = 1.0
    query_budget: int = 60_000
    max_query_terms: int = 96
    rrf_k: int = RRF_K
    #: pre-ranker pruning; 0 disables it (the raw union is then the candidate set)
    #:
    #: ``max`` binds only on a crowded entity — the average list is far shorter
    #: than it, and ``eps`` is what sets that average — so raising it costs
    #: candidates on exactly the entities that need them (a mall, a corporate
    #: park, a chain) and almost nothing elsewhere.
    prerank_max: int = 120
    prerank_min: int = 15
    #: Measured on the cached raw union, held-out fold 0, India (work/reports/
    #: probe_prune.json).  ``eps`` is the only one of the three that binds:
    #:
    #:   max/min/eps      cand/S1   pair recall   ceiling
    #:   60/15/0.002       29.4      0.9003       0.9595   <- the first full run
    #:   120/15/0.002      29.4      0.9003       0.9595   <- raising max does nothing
    #:   120/15/0.001      47.2      0.9090       0.9632
    #:   120/15/0.0005     81.6      0.9181       0.9672
    #:   raw union        159.8      0.9282       0.9715
    #:
    #: 0.001 buys +0.0037 of ceiling for 1.6x the pairs, which is what the
    #: per-pair stages of a run cost.  0.0005 buys +0.0077 for 2.8x, and is the
    #: next step once the matcher is shown to capture this one.
    prerank_eps: float = 0.001
    #: records kept per exact key.  A key matching thousands of records carries
    #: no information, but the old cap of 50 also discarded a real entity that
    #: happened to share a building with 50 others.
    key_block_cap: int = DEFAULT_KEY_BLOCK_CAP
    #: keep the *selective* keys' hits when a query has more than ``k_keys`` of
    #: them, instead of the lowest pool row ids.  Off by default because it
    #: changes the raw union, and a cached union must never mean two things — it
    #: gets its own fingerprint when on.
    key_select: bool = False
    tag: str = "b3"

    def specs(self) -> List[Tuple[str, IndexConfig]]:
        from dataclasses import replace
        return [(tag, replace(cfg,
                              max_df_abs=int(cfg.max_df_abs * self.df_scale),
                              max_df_frac=cfg.max_df_frac * self.df_scale,
                              query_budget=self.query_budget,
                              max_query_terms=self.max_query_terms))
                for tag, cfg in RETRIEVERS]

    def retrieval_fingerprint(self) -> str:
        """What decides the **raw** union, and nothing else.

        Split out from :meth:`fingerprint` so that changing a pruning threshold
        does not invalidate the raw candidate cache.  The raw union of one
        country partition costs an hour of retrieval and five gigabytes on disk;
        the pruning thresholds are the knobs most worth sweeping, and sweeping
        them used to mean re-running blocking for no reason.
        """
        parts = [f"{self.tag}_k{self.k_per_retriever}_u{self.max_union_per_source}",
                 f"df{self.df_scale:g}", f"qb{self.query_budget // 1000}k",
                 f"key{self.k_keys if self.use_keys else 0}"]
        if self.use_keys and self.key_block_cap != DEFAULT_KEY_BLOCK_CAP:
            parts.append(f"kb{self.key_block_cap}")
        if self.use_keys and self.key_select:
            parts.append("ks")
        return "_".join(parts)

    def fingerprint(self) -> str:
        """The raw union **and** the pruning that turns it into the scored set.

        ``prerank_min`` and ``prerank_eps`` belong in here: two configurations
        that differ only in ``eps`` produce different candidate sets, and before
        they were in the name one silently reused the other's shards.
        """
        return (f"{self.retrieval_fingerprint()}"
                f"_pr{self.prerank_max}-{self.prerank_min}-{self.prerank_eps:g}")


def _schema() -> pa.Schema:
    fields = [("s1", pa.int32()), ("src", pa.int8()), ("cid", pa.int32())]
    for tag in RETRIEVER_TAGS:
        fields.append((f"{tag}_score", pa.float32()))
        fields.append((f"{tag}_rank", pa.int16()))
    fields += [("key_hit", pa.int8()), ("rrf", pa.float32()), ("nret", pa.int8())]
    return pa.schema(fields)


CAND_SCHEMA = _schema()
MISSING_RANK = np.int16(999)


# ------------------------------------------------------------------ indexes
def index_path(split: str, country: str, source: str, cfg: IndexConfig,
               bcfg: "BlockingConfig") -> Path:
    """Cache path.  The caps go in the name: two indexes built with different
    document-frequency limits are different indexes, and silently reusing one for
    the other was a real bug during development."""
    safe = country.replace("/", "_").replace(" ", "_")
    tag = f"a{cfg.max_df_abs // 1000}k_f{cfg.max_df_frac:g}_n{cfg.min_df}"
    return config.INDEX_DIR / f"{split}_{safe}_{source}_{cfg.name}_{tag}.pkl"


def get_index(split: str, country: str, source: str, cfg: IndexConfig,
              bcfg: "BlockingConfig", pool: Partition | None = None,
              cache: bool = True, verbose: bool = True) -> SparseIndex:
    path = index_path(split, country, source, cfg, bcfg)
    if cache and path.exists():
        idx = SparseIndex.load(path)
        idx.cfg = cfg
        return idx
    pool = pool or load_partition(split, source, country)
    texts = [make_view(cfg, n, a) for n, a in zip(pool.names, pool.addrs)]
    idx = SparseIndex(cfg).build(texts, verbose=verbose)
    del texts
    gc.collect()
    if cache:
        idx.save(path)
    return idx


# ------------------------------------------------------------------ fusion
def fuse(nq: int, per_retriever: Sequence[Tuple[np.ndarray, np.ndarray]],
         key_hits: List[np.ndarray] | None, max_per_source: int,
         rrf_k: int = RRF_K) -> Dict[str, np.ndarray]:
    """Union the ranked lists of one (query chunk, source) into flat arrays."""
    n_ret = len(per_retriever)
    q_parts, c_parts, w_parts, r_parts, s_parts = [], [], [], [], []
    for li, (I, S) in enumerate(per_retriever):
        k = I.shape[1]
        keep = (I >= 0).reshape(-1)
        if not keep.any():
            continue
        q_parts.append(np.repeat(np.arange(nq, dtype=np.int64), k)[keep])
        c_parts.append(I.reshape(-1)[keep].astype(np.int64))
        w_parts.append(np.full(int(keep.sum()), li, dtype=np.int8))
        r_parts.append(np.tile(np.arange(k, dtype=np.int16), nq)[keep])
        s_parts.append(S.reshape(-1)[keep])
    if key_hits is not None and any(len(h) for h in key_hits):
        qs = np.concatenate([np.full(len(h), i, dtype=np.int64)
                             for i, h in enumerate(key_hits) if len(h)])
        cs = np.concatenate([h for h in key_hits if len(h)]).astype(np.int64)
        q_parts.append(qs); c_parts.append(cs)
        w_parts.append(np.full(len(cs), n_ret, dtype=np.int8))
        r_parts.append(np.zeros(len(cs), dtype=np.int16))
        s_parts.append(np.zeros(len(cs), dtype=np.float32))

    empty = {"q": np.zeros(0, np.int32), "cid": np.zeros(0, np.int32),
             "key_hit": np.zeros(0, np.int8), "rrf": np.zeros(0, np.float32),
             "nret": np.zeros(0, np.int8)}
    for tag in RETRIEVER_TAGS:
        empty[f"{tag}_score"] = np.zeros(0, np.float32)
        empty[f"{tag}_rank"] = np.zeros(0, np.int16)
    if not q_parts:
        return empty

    q = np.concatenate(q_parts); c = np.concatenate(c_parts)
    which = np.concatenate(w_parts); rank = np.concatenate(r_parts)
    score = np.concatenate(s_parts)
    del q_parts, c_parts, w_parts, r_parts, s_parts

    n_pool = int(c.max()) + 1
    uniq, inv = np.unique(q * n_pool + c, return_inverse=True)
    m = len(uniq)
    cols: Dict[str, np.ndarray] = {}
    for tag in RETRIEVER_TAGS:
        cols[f"{tag}_score"] = np.zeros(m, dtype=np.float32)
        cols[f"{tag}_rank"] = np.full(m, MISSING_RANK, dtype=np.int16)
    key_hit = np.zeros(m, dtype=np.int8)
    rrf = np.zeros(m, dtype=np.float32)
    nret = np.zeros(m, dtype=np.int8)

    for li in range(n_ret + 1):
        sel = which == li
        if not sel.any():
            continue
        idx = inv[sel]
        if li < n_ret:
            tag = RETRIEVER_TAGS[li]
            np.maximum.at(cols[f"{tag}_score"], idx, score[sel])
            np.minimum.at(cols[f"{tag}_rank"], idx, rank[sel])
            np.add.at(rrf, idx, (1.0 / (rrf_k + rank[sel] + 1)).astype(np.float32))
            np.add.at(nret, idx, 1)
        else:
            key_hit[idx] = 1

    qq = (uniq // n_pool).astype(np.int32)
    cc = (uniq % n_pool).astype(np.int32)
    if max_per_source > 0 and m > 0:
        rank_in_q = _rank_within(qq, rrf)
        sel = np.flatnonzero(rank_in_q < max_per_source)
        qq, cc = qq[sel], cc[sel]
        key_hit, rrf, nret = key_hit[sel], rrf[sel], nret[sel]
        for k in list(cols):
            cols[k] = cols[k][sel]
    out = {"q": qq, "cid": cc, "key_hit": key_hit, "rrf": rrf, "nret": nret}
    out.update(cols)
    return out


def _first_unique(rows: np.ndarray, k: int) -> np.ndarray:
    """First ``k`` distinct values, **in the order given**.

    ``np.unique`` sorts, which would throw away the selectivity ordering
    :meth:`ber.keys.KeyBlocks.lookup` puts the hits in and truncate by pool index
    instead — silently keeping the low-numbered records of a huge block over the
    whole of a tiny one.
    """
    _, idx = np.unique(rows, return_index=True)
    return rows[np.sort(idx)[:k]]


def _rank_within(group: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each row inside its (unsorted) group, best score first."""
    order = np.lexsort((-score, group))
    g = group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    pos = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    out = np.empty(len(group), dtype=np.int32)
    out[order] = pos
    return out


# ------------------------------------------------------------------ driver
def cand_dir(split: str, country: str, bcfg: BlockingConfig,
             pruned: bool = False) -> Path:
    """Where a partition's candidates live.

    ``pruned=False`` is the raw retriever union; ``pruned=True`` is what the
    pre-ranker kept, which is the set the matcher scores and therefore the set
    written to ``candidate_pairs.tsv``.
    """
    safe = country.replace("/", "_").replace(" ", "_")
    fp = bcfg.fingerprint() if pruned else bcfg.retrieval_fingerprint()
    sub = "pruned" if pruned else "raw"
    return config.CAND_DIR / fp / sub / f"{split}_{safe}"


def _query_stamp(q_ids: np.ndarray | None) -> str:
    """Fingerprint of the query set, so a cached partition is only reused for it."""
    import hashlib
    if q_ids is None:
        return "ALL"
    a = np.sort(np.asarray(q_ids, dtype=np.int32))
    return f"{len(a)}:{hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest()}"


def retrieve_partition(split: str, country: str, bcfg: BlockingConfig,
                       roles: TokenRoles, q_ids: np.ndarray | None = None,
                       chunk: int = 20_000, verbose: bool = True,
                       cache_index: bool = True, force: bool = False) -> Path:
    """Generate the raw candidate union for one (split, country)."""
    out = cand_dir(split, country, bcfg)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _query_stamp(q_ids)
    done = out / "_DONE"
    if done.exists() and not force:
        if done.read_text().strip() == stamp:
            return out
        # a different query set was cached here (e.g. a smoke run): start over
        for f in out.glob("*.parquet"):
            f.unlink()
        done.unlink()

    s1 = load_partition(split, "S1", country)
    rows = (np.flatnonzero(np.isin(s1.ids, np.asarray(q_ids, dtype=np.int32)))
            if q_ids is not None else np.arange(len(s1), dtype=np.int64))
    q_ids_arr = s1.ids[rows]
    q_names = [s1.names[i] for i in rows]
    q_addrs = [s1.addrs[i] for i in rows]
    del s1
    gc.collect()
    nq = len(rows)
    if verbose:
        print(f"[retrieve] {split}/{country}: {nq:,} queries", flush=True)

    specs = bcfg.specs()
    for source in ("S2", "S3"):
        shard = out / f"{source}.parquet"
        if shard.exists() and not force:
            if verbose:
                print(f"  [retrieve] {source}: cached", flush=True)
            continue
        t0 = time.time()
        # The pool text is ~0.6 GB of Python strings for a 2M-record partition and
        # is only needed to *build* an index or the key blocks.  Holding it through
        # the query loop pushed the working set past what this machine keeps
        # resident, and the sparse product then ran at a fraction of its speed.
        pool = load_partition(split, source, country)
        pool_ids = pool.ids.copy()
        kb = (KeyBlocks.build(pool.names, pool.addrs, country, roles,
                              max_block=bcfg.key_block_cap)
              if bcfg.use_keys else None)
        need_build = [cfg for _, cfg in specs
                      if not index_path(split, country, source, cfg, bcfg).exists()]
        for cfg in need_build:
            get_index(split, country, source, cfg, bcfg, pool, cache_index, verbose)
            gc.collect()
        del pool
        gc.collect()

        # one index in memory at a time: query every chunk against it, keep the hits
        hits: List[Tuple[np.ndarray, np.ndarray]] = []
        for tag, cfg in specs:
            idx = get_index(split, country, source, cfg, bcfg, None, cache_index, verbose)
            I = np.empty((nq, bcfg.k_per_retriever), dtype=np.int32)
            S = np.empty((nq, bcfg.k_per_retriever), dtype=np.float32)
            t_r = time.time()
            for lo in range(0, nq, chunk):
                hi = min(lo + chunk, nq)
                qt = [make_view(cfg, n, a)
                      for n, a in zip(q_names[lo:hi], q_addrs[lo:hi])]
                i, s = idx.query(qt, k=bcfg.k_per_retriever)
                I[lo:hi], S[lo:hi] = i, s
                if verbose:
                    rate = hi / max(time.time() - t_r, 1e-9)
                    print(f"      [{tag}] {hi:,}/{nq:,} queries "
                          f"({rate:,.0f}/s, eta {(nq - hi) / max(rate, 1e-9) / 60:.1f} min)",
                          flush=True)
            hits.append((I, S))
            if verbose:
                print(f"    [{tag}] done in {time.time() - t_r:.0f}s "
                      f"({1000 * (time.time() - t_r) / nq:.2f} s/1k)", flush=True)
            del idx
            gc.collect()

        writer = pq.ParquetWriter(shard.with_suffix(".tmp"), CAND_SCHEMA,
                                  compression="zstd", compression_level=3)
        src_code = np.int8(2 if source == "S2" else 3)
        n_rows = 0
        for lo in range(0, nq, chunk):
            hi = min(lo + chunk, nq)
            per = [(I[lo:hi], S[lo:hi]) for I, S in hits]
            khits = None
            if kb is not None:
                khits = []
                for n, a in zip(q_names[lo:hi], q_addrs[lo:hi]):
                    h = kb.lookup(record_keys(n, a, country, roles),
                                  by_selectivity=bcfg.key_select)
                    khits.append(_first_unique(h, bcfg.k_keys) if len(h) else h)
            t = fuse(hi - lo, per, khits, bcfg.max_union_per_source, bcfg.rrf_k)
            if len(t["q"]):
                data = {"s1": pa.array(q_ids_arr[lo + t["q"]], pa.int32()),
                        "src": pa.array(np.full(len(t["q"]), src_code), pa.int8()),
                        "cid": pa.array(pool_ids[t["cid"]], pa.int32())}
                for tag in RETRIEVER_TAGS:
                    data[f"{tag}_score"] = pa.array(t[f"{tag}_score"], pa.float32())
                    data[f"{tag}_rank"] = pa.array(t[f"{tag}_rank"], pa.int16())
                data["key_hit"] = pa.array(t["key_hit"], pa.int8())
                data["rrf"] = pa.array(t["rrf"], pa.float32())
                data["nret"] = pa.array(t["nret"], pa.int8())
                writer.write_table(pa.table(data, schema=CAND_SCHEMA))
                n_rows += len(t["q"])
        writer.close()
        shard.with_suffix(".tmp").replace(shard)
        del hits, kb, pool_ids
        gc.collect()
        if verbose:
            print(f"  [retrieve] {source}: {n_rows:,} pairs in {time.time() - t0:.0f}s",
                  flush=True)

    (out / "meta.json").write_text(json.dumps(
        {"split": split, "country": country, "n_queries": int(nq),
         "config": asdict(bcfg)}, indent=1))
    done.write_text(stamp + "\n")
    return out


def load_candidates(split: str, country: str, bcfg: BlockingConfig,
                    columns: Sequence[str] | None = None,
                    pruned: bool = False,
                    s1_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Read a partition's candidate shards.

    ``s1_ids`` filters **per row group**, before anything is concatenated: a raw
    country table is ~60M rows and materialising all of it only to keep a
    seventh of it costs gigabytes for no reason.
    """
    d = cand_dir(split, country, bcfg, pruned=pruned)
    parts = sorted(d.glob("S*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no candidate shards in {d}")
    cols = list(columns) if columns else None
    if s1_ids is None:
        return pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in parts],
                         ignore_index=True)
    want = np.asarray(s1_ids, dtype=np.int32)
    out: List[pd.DataFrame] = []
    for part in parts:
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=2_000_000, columns=cols):
            df = batch.to_pandas()
            m = np.isin(df["s1"].to_numpy(), want)
            if m.any():
                out.append(df.loc[m].reset_index(drop=True))
            del df
        del pf
    if not out:
        return pd.DataFrame(columns=cols or [f.name for f in CAND_SCHEMA])
    return pd.concat(out, ignore_index=True)


def candidates_as_dict(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for a, b, c in zip(df["s1"].to_numpy(), df["src"].to_numpy(), df["cid"].to_numpy()):
        out.setdefault(f"S1-{a}", []).append(f"S{b}-{c}")
    return out


#: The configuration the submission is built with (see PROGRESS.md / RESULTS.md).
DEFAULT_BLOCKING = BlockingConfig()
