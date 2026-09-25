"""The pre-ranker: the last filter before the matcher, and therefore the file.

``candidate_pairs.tsv`` is defined as "the exact set of records you feed into
your matching model for inference".  This LightGBM sits on the retrieval columns
alone — each retriever's score and rank, the RRF fusion, whether a key block fired
and how many retrievers agreed — so it costs nothing beyond what blocking already
produced, and its output *is* the candidate set.

Pruning is adaptive, as the research doc prescribes: keep every candidate above
``eps``, but never fewer than ``min_keep`` and never more than ``max_keep`` per
S1 entity.  A hard top-k would throw away the tail of a genuinely large cluster;
a pure threshold would leave a generic-name entity with hundreds of candidates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .model import TrainedModel, train_lgb
from .retrieval import RETRIEVAL_COLUMNS

#: ``pre`` is what this model produces, so it is not one of its inputs.
PRERANK_FEATURES = [c for c in RETRIEVAL_COLUMNS if c != "pre"]

PRERANK_PARAMS = dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=500)


def matrix(df: pd.DataFrame) -> np.ndarray:
    M = np.empty((len(df), len(PRERANK_FEATURES)), dtype=np.float32)
    for j, c in enumerate(PRERANK_FEATURES):
        M[:, j] = df[c].to_numpy(dtype=np.float32)
    return M


def train_preranker(df: pd.DataFrame, y: np.ndarray,
                    holdout: np.ndarray | None = None) -> TrainedModel:
    X = matrix(df)
    if holdout is None:
        return train_lgb(X, y, PRERANK_FEATURES, params=PRERANK_PARAMS,
                         num_boost_round=400, early_stopping=0, monotone=True)
    return train_lgb(X[~holdout], y[~holdout], PRERANK_FEATURES,
                     X[holdout], y[holdout], params=PRERANK_PARAMS,
                     num_boost_round=1500, early_stopping=60, monotone=True)


def prune(df: pd.DataFrame, prob: np.ndarray, max_keep: int, min_keep: int,
          eps: float) -> np.ndarray:
    """Boolean mask of the candidates the matcher will score.

    Rows may be in any order; grouping is done on ``s1``.
    """
    s1 = df["s1"].to_numpy()
    order = np.lexsort((-prob, s1))
    g = s1[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    rank = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    p_sorted = prob[order]
    keep_sorted = (rank < min_keep) | ((p_sorted >= eps) & (rank < max_keep))
    keep = np.empty(len(prob), dtype=bool)
    keep[order] = keep_sorted
    return keep


def model_path(bcfg=None) -> Path:
    """Cache path, keyed by the **retrieval** configuration.

    The pre-ranker is trained on one blocking configuration's candidate table and
    is meaningless on another. Keying only on the filename meant a run silently
    reused a pre-ranker fitted to a different retriever set — caught before it
    reached a scored submission, but only just.

    It keys on the retrieval fingerprint rather than the full one because its
    inputs are the raw union's retrieval columns and its target is the ground
    truth: neither changes when a pruning threshold does, and this model is what
    *applies* those thresholds, so refitting it for each one would be circular as
    well as wasteful.
    """
    tag = bcfg.retrieval_fingerprint() if bcfg is not None else "default"
    return config.MODEL_DIR / f"preranker_{tag}.pkl"


def prerank_partition(split: str, country: str, bcfg, model: TrainedModel,
                      chunk_rows: int = 8_000_000, verbose: bool = True,
                      force: bool = False) -> Path:
    """Score the raw union, prune it, and write the pruned candidate shards.

    The pruned shards carry a ``pre`` column so the matcher can use the
    pre-ranker's opinion as a feature.
    """
    import gc
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    from .retrieval import CAND_SCHEMA, cand_dir

    src_dir = cand_dir(split, country, bcfg, pruned=False)
    out = cand_dir(split, country, bcfg, pruned=True)
    out.mkdir(parents=True, exist_ok=True)
    stamp = (src_dir / "_DONE").read_text().strip() if (src_dir / "_DONE").exists() else "?"
    done = out / "_DONE"
    if done.exists() and not force:
        if done.read_text().strip() == stamp:
            return out
        for f in out.glob("*.parquet"):
            f.unlink()
        done.unlink()

    schema = pa.schema(list(CAND_SCHEMA) + [pa.field("pre", pa.float32())])
    t0 = time.time()
    kept = total = 0
    for shard in sorted(src_dir.glob("S*.parquet")):
        writer = pq.ParquetWriter(out / f"{shard.stem}.tmp", schema,
                                  compression="zstd", compression_level=3)
        pf = pq.ParquetFile(shard)
        buf = []
        n_buf = 0

        def flush(buf):
            if not buf:
                return 0, 0
            df = pd.concat(buf, ignore_index=True)
            p = model.raw_predict(matrix(df))
            keep = prune(df, p, bcfg.prerank_max, bcfg.prerank_min, bcfg.prerank_eps)
            sub = df.loc[keep].copy()
            sub["pre"] = p[keep].astype(np.float32)
            writer.write_table(pa.Table.from_pandas(sub, schema=schema,
                                                    preserve_index=False))
            return len(df), int(keep.sum())

        # rows of one S1 must not be split across flushes, so cut on an S1 boundary
        for batch in pf.iter_batches(batch_size=1_000_000):
            df = batch.to_pandas()
            buf.append(df)
            n_buf += len(df)
            if n_buf >= chunk_rows:
                merged = pd.concat(buf, ignore_index=True)
                last = merged["s1"].iat[-1]
                head = merged[merged["s1"] != last]
                tail = merged[merged["s1"] == last]
                t, k = flush([head])
                total += t; kept += k
                buf = [tail]; n_buf = len(tail)
        t, k = flush(buf)
        total += t; kept += k
        writer.close()
        (out / f"{shard.stem}.tmp").replace(out / f"{shard.stem}.parquet")
        del pf
        gc.collect()
    done.write_text(stamp + "\n")
    if verbose:
        print(f"[prerank] {split}/{country}: {total:,} -> {kept:,} pairs "
              f"({kept / max(total, 1):.1%}) in {time.time() - t0:.0f}s", flush=True)
    return out
