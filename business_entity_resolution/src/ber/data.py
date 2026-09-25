"""Cached, memory-aware access to the challenge files.

The TSVs are converted once into parquet (``work/cache``).  Entity ids are stored
as int32 (``S2-166376419`` -> ``166376419``, max observed 999,999,995) because
five million Python strings cost ~300 MB where the numbers cost 20 MB; the
prefix is implied by the source.

A :class:`Partition` is the unit every stage works on: the records of one
``(split, source, country)`` triple, already normalised into the views Module 2
defines.  Country is the exact string from the file — an open set, never a list
we hard-code.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .io_utils import READ_KW, SOURCE_COLS

ID_PREFIX = {"S1": "S1-", "S2": "S2-", "S3": "S3-"}


# ------------------------------------------------------------------ cache
def cache_path(split: str, source: str) -> Path:
    return config.CACHE_DIR / f"{split}_{source}.parquet"


def build_cache(split: str, source: str, force: bool = False,
                chunksize: int = 1_000_000) -> Path:
    """Convert one source TSV to parquet.  Resumable: skips an existing file."""
    out = cache_path(split, source)
    if out.exists() and not force:
        return out
    src = config.FILES[(split, source)]
    tmp = out.with_suffix(".parquet.tmp")
    writer = None
    schema = pa.schema([
        ("id", pa.int32()),
        ("name", pa.string()),
        ("addr", pa.string()),
        ("country", pa.dictionary(pa.int8(), pa.string())),
    ])
    n = 0
    for chunk in pd.read_csv(src, chunksize=chunksize, **READ_KW):
        chunk = chunk[SOURCE_COLS]
        ids = chunk["entity_id"].str.slice(3).astype("int64").to_numpy().astype(np.int32)
        tbl = pa.table({
            "id": pa.array(ids, pa.int32()),
            "name": pa.array(chunk["business_name"].tolist(), pa.string()),
            "addr": pa.array(chunk["business_address"].tolist(), pa.string()),
            "country": pa.array(chunk["country"].tolist(), pa.string())
                         .dictionary_encode().cast(pa.dictionary(pa.int8(), pa.string())),
        }, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3)
        writer.write_table(tbl)
        n += len(chunk)
        del chunk, tbl
    if writer is not None:
        writer.close()
    tmp.replace(out)
    return out


def build_all_caches(force: bool = False) -> Dict[Tuple[str, str], int]:
    sizes = {}
    for (split, source) in config.FILES:
        t0 = time.time()
        p = build_cache(split, source, force=force)
        sizes[(split, source)] = p.stat().st_size
        gc.collect()
    return sizes


def load_frame(split: str, source: str, columns: Sequence[str] | None = None,
               country: str | None = None) -> pd.DataFrame:
    """Load a cached source, optionally one country only."""
    build_cache(split, source)
    filters = [("country", "==", country)] if country else None
    tbl = pq.read_table(cache_path(split, source), columns=list(columns) if columns else None,
                        filters=filters)
    return tbl.to_pandas()


def countries(split: str, source: str) -> List[str]:
    """Distinct country labels, sorted — discovered from the data, never listed."""
    df = load_frame(split, source, columns=["country"])
    return sorted(df["country"].astype(str).unique().tolist())


# ------------------------------------------------------------------ records
@dataclass
class Partition:
    """Records of one (split, source, country) triple."""
    split: str
    source: str
    country: str
    ids: np.ndarray                      # int32 numeric part of the entity id
    names: List[str]
    addrs: List[str]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def prefix(self) -> str:
        return ID_PREFIX[self.source]

    def entity_ids(self, idx: Iterable[int] | None = None) -> List[str]:
        pre = self.prefix
        arr = self.ids if idx is None else self.ids[np.asarray(idx, dtype=np.int64)]
        return [pre + str(int(i)) for i in arr]

    def entity_id(self, i: int) -> str:
        return self.prefix + str(int(self.ids[i]))


def load_partition(split: str, source: str, country: str,
                   limit: int | None = None) -> Partition:
    df = load_frame(split, source, columns=["id", "name", "addr"], country=country)
    if limit is not None and len(df) > limit:
        df = df.iloc[:limit]
    return Partition(split, source, country,
                     df["id"].to_numpy(dtype=np.int32),
                     df["name"].tolist(), df["addr"].tolist())


def load_s1(split: str, country: str | None = None) -> Partition:
    return load_partition(split, "S1", country) if country else _load_all(split, "S1")


def _load_all(split: str, source: str) -> Partition:
    df = load_frame(split, source, columns=["id", "name", "addr", "country"])
    return Partition(split, source, "*", df["id"].to_numpy(dtype=np.int32),
                     df["name"].tolist(), df["addr"].tolist())


def s1_country_map(split: str) -> pd.DataFrame:
    """``id`` (int32) and ``country`` for every S1 record of a split, file order."""
    return load_frame(split, "S1", columns=["id", "country"])


# ------------------------------------------------------------------ labels
def load_gt_arrays(gt_path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground truth as three aligned arrays: ``s1_id``, ``match_id``, ``match_src``.

    ``match_src`` is 2 or 3.  Singletons contribute no row, so use
    :func:`gt_sizes` when the per-entity count matters.
    """
    path = gt_path or config.GROUND_TRUTH
    s1_list, m_list, src_list = [], [], []
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                continue
            rest = rest.strip()
            if not rest:
                continue
            sid = int(s1[3:])
            for tok in rest.split(","):
                if not tok:
                    continue
                s1_list.append(sid)
                m_list.append(int(tok[3:]))
                src_list.append(2 if tok[1] == "2" else 3)
    return (np.asarray(s1_list, dtype=np.int32),
            np.asarray(m_list, dtype=np.int32),
            np.asarray(src_list, dtype=np.int8))


def load_gt_dict(gt_path: Path | None = None) -> Dict[str, List[str]]:
    from .io_utils import read_ground_truth
    return read_ground_truth(gt_path)
