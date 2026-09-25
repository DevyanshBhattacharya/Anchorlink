"""Leak-free validation splits.

Two splits, both described in Module 6 of the research doc:

* **Name-family GroupKFold.**  The unit is one S1 entity plus its matches, which
  are disjoint by construction (no S2/S3 record belongs to two S1 entities — 0
  violations in 7,638,365 training pairs).  That alone is not enough: 35–44% of
  S1 names repeat, so two *different* businesses with the same name would land in
  different folds and the model could memorise the name.  Clusters are therefore
  unioned into **name families** by their core-name skeleton before folding.
* **Leave-one-country-out.**  Train on US, score India, and the reverse.  France
  has no training rows, so this cross-country gap is the only honest proxy for it.

Fold assignment is a deterministic hash of the family key, so it is stable across
runs and machines and needs no stored state beyond the cached table.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .data import load_frame
from .normalize import TokenRoles, skeleton, tokens

N_FOLDS = 5


def _family_key(name: str, country: str, roles: TokenRoles | None) -> str:
    """Core-name skeleton, sorted — the key that unions look-alike clusters."""
    core = roles.core_tokens(name, country) if roles is not None else tokens(name)
    sk = skeleton(" ".join(core))
    parts = sorted(p for p in sk.split() if p)
    if not parts:
        parts = sorted(p for p in skeleton(name).split() if p)
    return " ".join(parts) or "<empty>"


def build_name_families(split: str = "train", roles: TokenRoles | None = None,
                        force: bool = False) -> pd.DataFrame:
    """``id``, ``country``, ``family`` (str) and ``fold`` for every S1 record."""
    out = config.CACHE_DIR / f"{split}_s1_families.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    df = load_frame(split, "S1", columns=["id", "name", "country"])
    ctry = df["country"].astype(str).tolist()
    names = df["name"].tolist()
    fam = [_family_key(n, c, roles) for n, c in zip(names, ctry)]
    fold = np.fromiter(
        (zlib.crc32(f.encode("utf-8")) % N_FOLDS for f in fam),
        dtype=np.int8, count=len(fam))
    res = pd.DataFrame({"id": df["id"].to_numpy(dtype=np.int32),
                        "country": ctry, "family": fam, "fold": fold})
    res.to_parquet(out, index=False, compression="zstd")
    return res


# ------------------------------------------------------------------ selectors
def fold_ids(fam: pd.DataFrame, fold: int, country: str | None = None,
             invert: bool = False) -> np.ndarray:
    m = (fam["fold"].to_numpy() != fold) if invert else (fam["fold"].to_numpy() == fold)
    if country is not None:
        m &= (fam["country"].to_numpy() == country)
    return fam["id"].to_numpy(dtype=np.int32)[m]


def country_ids(fam: pd.DataFrame, country: str) -> np.ndarray:
    return fam["id"].to_numpy(dtype=np.int32)[fam["country"].to_numpy() == country]


def sample_ids(ids: np.ndarray, n: int, seed: int = config.SEED) -> np.ndarray:
    """Deterministic subsample, sorted so downstream joins stay stable."""
    if n >= len(ids):
        return np.sort(ids)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=n, replace=False))


def sample_by_family(fam: pd.DataFrame, mask: np.ndarray, n: int,
                     seed: int = config.SEED) -> np.ndarray:
    """Subsample **whole name families**, not individual S1 entities.

    Competition for an S2/S3 record comes overwhelmingly from S1 entities with a
    similar name, which is exactly what a family is.  Sampling entities one by
    one would break those groups apart and make the exclusivity features look
    far less contested in validation than they are at test time, where every S1
    queries.  Sampling families keeps the contest intact.
    """
    ids = fam["id"].to_numpy(dtype=np.int32)[mask]
    if n >= len(ids):
        return np.sort(ids)
    families = fam["family"].to_numpy()[mask]
    uniq, inv = np.unique(families, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[order] = np.arange(len(uniq))
    sizes = np.bincount(inv, minlength=len(uniq))
    cum = np.cumsum(sizes[order])
    n_fam = int(np.searchsorted(cum, n, side="left")) + 1
    keep_fam = np.zeros(len(uniq), dtype=bool)
    keep_fam[order[:n_fam]] = True
    return np.sort(ids[keep_fam[inv]])


def validation_plan(fam: pd.DataFrame, fold: int = 0,
                    per_country: int | None = None,
                    seed: int = config.SEED) -> Dict[str, np.ndarray]:
    """``{country: s1 ids}`` for the held-out fold, optionally subsampled."""
    plan: Dict[str, np.ndarray] = {}
    for c in sorted(fam["country"].astype(str).unique()):
        ids = fold_ids(fam, fold, country=c)
        plan[c] = sample_ids(ids, per_country, seed) if per_country else np.sort(ids)
    return plan


def training_plan(fam: pd.DataFrame, fold: int = 0,
                  per_country: int | None = None,
                  seed: int = config.SEED) -> Dict[str, np.ndarray]:
    """``{country: s1 ids}`` for everything outside the held-out fold."""
    plan: Dict[str, np.ndarray] = {}
    for c in sorted(fam["country"].astype(str).unique()):
        ids = fold_ids(fam, fold, country=c, invert=True)
        plan[c] = sample_ids(ids, per_country, seed + 1) if per_country else np.sort(ids)
    return plan


def loco_pairs(fam: pd.DataFrame) -> List[Tuple[str, str]]:
    """``(train_country, eval_country)`` pairs for leave-one-country-out."""
    cs = sorted(fam["country"].astype(str).unique())
    return [(a, b) for a in cs for b in cs if a != b]


def test_mix_weights(train_countries: Sequence[str],
                     split: str = "test") -> Dict[str, float]:
    """Country weights read off the test split, plus a LOCO weight.

    Every test country that also appears in training gets its own share of the
    test S1 entities; the share of every test country that does **not** appear in
    training is pooled into ``LOCO``, because the leave-one-country-out fold is
    the only estimate we have for it.  On the real files this returns
    ``{India: 0.467, US: 0.383, LOCO: 0.150}`` — the mix quoted in the research
    doc — without naming a single country in the code.
    """
    from .data import load_frame
    counts = (load_frame(split, "S1", columns=["country"])["country"]
              .astype(str).value_counts())
    total = float(counts.sum())
    if total <= 0:
        return {"LOCO": 1.0}
    known = set(train_countries)
    out: Dict[str, float] = {}
    unseen = 0.0
    for c, n in counts.items():
        if c in known:
            out[c] = n / total
        else:
            unseen += n / total
    if unseen > 0:
        out["LOCO"] = unseen
    return out


def reweighted_score(by_country: Dict[str, float], loco: float,
                     mix: Dict[str, float] | None = None) -> float:
    """The single headline number: each country weighted by its test share.

    Countries missing from ``by_country`` drop out and the remaining weights are
    renormalised, so the formula still works on a partial run.  With no mix and
    no usable weights it falls back to the plain mean of what it was given.
    """
    if mix is None:
        mix = dict(config.TEST_MIX)
    total = w = 0.0
    for c, weight in mix.items():
        v = loco if c == "LOCO" else by_country.get(c)
        if v is None or not np.isfinite(v):
            continue
        total += weight * v
        w += weight
    if w:
        return total / w
    vals = [v for v in by_country.values() if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
