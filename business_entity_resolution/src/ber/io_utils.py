"""Safe TSV loaders and validator-safe writers.

Two rules the rest of the code depends on:

* Every read uses ``sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
  keep_default_na=False``.  Business names such as ``NA``, ``NaN``, ``None`` and
  ``NULL`` are real names in this dataset; the pandas defaults would turn them
  into missing values.
* Every write uses ``open(..., newline="\n")`` and plain string formatting.
  ``csv.writer`` defaults to CRLF, which silently appends ``\r`` to the last ID
  of every row: the validator's prefix check still passes while the scorer loses
  that match.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd

from . import config

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

READ_KW = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
               keep_default_na=False, na_filter=False, encoding="utf-8")


# ------------------------------------------------------------------ readers
def read_source(split: str, source: str, nrows: int | None = None) -> pd.DataFrame:
    """Load ``{split}_{source}.tsv`` with entity_id/business_name/business_address/country."""
    path = config.FILES[(split, source)]
    df = pd.read_csv(path, nrows=nrows, **READ_KW)
    missing = [c for c in SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    return df[SOURCE_COLS]


def read_tsv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read any challenge-format TSV with the safe keyword set."""
    return pd.read_csv(Path(path), nrows=nrows, **READ_KW)


def read_ground_truth(path: str | Path | None = None) -> Dict[str, List[str]]:
    """``{source1_entity_id: [matched ids]}`` for every row, singletons included."""
    path = Path(path) if path is not None else config.GROUND_TRUTH
    gt: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header[:2] != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in fh:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab and not s1:
                continue
            rest = rest.strip()
            gt[s1] = [x for x in rest.split(",") if x] if rest else []
    return gt


def read_id_list(path: str | Path) -> Dict[str, List[str]]:
    """Read either ``matched_entity_ids`` or ``candidate_entity_ids`` shaped files."""
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # header, either flavour
        for line in fh:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab and not s1:
                continue
            rest = rest.strip()
            out[s1] = [x for x in rest.split(",") if x] if rest else []
    return out


def read_s1_ids(split: str) -> List[str]:
    """Ordered list of S1 entity ids for a split (the submission row order)."""
    path = config.FILES[(split, "S1")]
    ids: List[str] = []
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            if line.strip():
                ids.append(line.split("\t", 1)[0].strip())
    return ids


def iter_source(split: str, source: str, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream a source file in chunks, for memory-bounded passes."""
    path = config.FILES[(split, source)]
    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        yield chunk[SOURCE_COLS]


# ------------------------------------------------------------------ writers
_ALLOWED_PREFIX = ("S2-", "S3-")


def write_id_list_tsv(path: str | Path,
                      header: Sequence[str],
                      mapping: Dict[str, Iterable[str]],
                      all_s1: Sequence[str]) -> Tuple[int, int, int]:
    """Write one row per S1 with LF endings, no quoting and no spaces.

    Returns ``(rows, non_empty_rows, total_ids)``.  IDs are de-duplicated in
    first-seen order and filtered to the S2-/S3- prefixes; anything else would be
    rejected by the scorer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = non_empty = total = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for s1 in all_s1:
            ids = list(dict.fromkeys(
                i for i in mapping.get(s1, ()) if i[:3] in _ALLOWED_PREFIX
            ))
            rows += 1
            if ids:
                non_empty += 1
                total += len(ids)
            fh.write(f"{s1}\t{','.join(ids)}\n")
    return rows, non_empty, total


def write_matching(path: str | Path, mapping, all_s1):
    return write_id_list_tsv(path, ["source1_entity_id", "matched_entity_ids"],
                             mapping, all_s1)


def write_candidates(path: str | Path, mapping, all_s1):
    return write_id_list_tsv(path, ["source1_entity_id", "candidate_entity_ids"],
                             mapping, all_s1)
