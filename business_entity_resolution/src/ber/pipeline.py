"""Phase-0 orchestration: block -> featurise -> label -> train -> decide -> write.

Fold discipline (all folds are name-family GroupKFold, per country):

* folds 1–2  train M1 (the pairwise/retrieval model)
* fold 3     trains the **pre-ranker**.  It has its own fold because the
  pre-ranker's score is an M1 feature: trained on M1's own rows it would be an
  in-sample prediction, and M1 would over-trust it — exactly the feature that
  behaves differently at test time, where it is always out-of-sample.
* fold 4a    trains M2 (the meta model, which needs M1's out-of-sample scores)
* fold 4b    fits isotonic calibration on M2's out-of-sample scores
* fold 0     is validation, never seen by any fit, and its queries search the
  **full** S2/S3 pool so the distractor density matches test

The leave-one-country-out variant does the same with the fold roles played by
countries, which is the only honest proxy for France.
"""
from __future__ import annotations

import gc
import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .corpus import build_roles, roles_path
from .data import load_partition
from .decide import decide, exclusivity_arrays, temper
from .features import (META_FEATURES, PAIR_FEATURES, build_meta_features,
                       group_starts)
from .featurize import featurize_partition, load_features
from .io_utils import write_candidates, write_matching, read_s1_ids
from .labels import GroundTruth
from .model import (TrainedModel, expected_calibration_error, fit_calibrator,
                    reliability_table, train_lgb)
from .retrieval import BlockingConfig, load_candidates, retrieve_partition
from .scoring import f05_ceiling, macro_f05, macro_f05_breakdown
from .splits import build_name_families, fold_ids, sample_ids

STAGE1_FEATURES = list(PAIR_FEATURES)
STAGE2_FEATURES = list(PAIR_FEATURES) + list(META_FEATURES)


# ------------------------------------------------------------------ helpers
def sub_fold(ids: np.ndarray, part: int, n_parts: int = 2) -> np.ndarray:
    """Deterministic split of one fold, used to keep calibration out-of-sample."""
    h = np.fromiter((zlib.crc32(str(int(i)).encode()) % n_parts for i in ids),
                    dtype=np.int8, count=len(ids))
    return ids[h == part]


@dataclass
class Selection:
    """Which S1 ids play which role, per country.  The five sets are disjoint."""
    m1: Dict[str, np.ndarray] = field(default_factory=dict)
    pre: Dict[str, np.ndarray] = field(default_factory=dict)
    m2: Dict[str, np.ndarray] = field(default_factory=dict)
    cal: Dict[str, np.ndarray] = field(default_factory=dict)
    val: Dict[str, np.ndarray] = field(default_factory=dict)

    def all_ids(self, country: str) -> np.ndarray:
        parts = [d.get(country) for d in (self.m1, self.pre, self.m2, self.cal, self.val)]
        parts = [p for p in parts if p is not None and len(p)]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int32)


def plan_selection(fam: pd.DataFrame, n_m1: int, n_m2: int, n_cal: int, n_val: int,
                   val_fold: int = 0, meta_fold: int = 4,
                   seed: int = config.SEED) -> Selection:
    sel = Selection()
    for c in sorted(fam["country"].astype(str).unique()):
        m1_ids = np.concatenate([fold_ids(fam, f, country=c) for f in (1, 2, 3)])
        meta_ids = fold_ids(fam, meta_fold, country=c)
        sel.m1[c] = sample_ids(m1_ids, n_m1, seed)
        sel.m2[c] = sample_ids(sub_fold(meta_ids, 0), n_m2, seed + 1)
        sel.cal[c] = sample_ids(sub_fold(meta_ids, 1), n_cal, seed + 2)
        sel.val[c] = sample_ids(fold_ids(fam, val_fold, country=c), n_val, seed + 3)
    return sel


def prepare(force: bool = False) -> Tuple[pd.DataFrame, object]:
    """Caches, corpus statistics and name families.  Idempotent."""
    roles = build_roles("train", force=force)
    build_roles("test", force=force)
    fam = build_name_families("train", roles=roles, force=force)
    return fam, roles


# ------------------------------------------------------------------ stages
def run_blocking(split: str, countries: Sequence[str], bcfg: BlockingConfig,
                 roles, q_by_country: Dict[str, np.ndarray] | None = None,
                 force: bool = False) -> None:
    for c in countries:
        retrieve_partition(split, c, bcfg, roles,
                           q_ids=None if q_by_country is None else q_by_country.get(c),
                           force=force)
        gc.collect()


def run_featurize(split: str, countries: Sequence[str], bcfg: BlockingConfig,
                  n_workers: int | None = None, block_s1: int = 4000,
                  force: bool = False) -> None:
    rp = roles_path(split)
    for c in countries:
        featurize_partition(split, c, bcfg, rp, block_s1=block_s1,
                            n_workers=n_workers, force=force)
        gc.collect()


def gather(split: str, countries: Sequence[str], bcfg: BlockingConfig,
           ids_by_country: Dict[str, np.ndarray] | None = None
           ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Stack feature shards for the given countries, optionally filtered by S1 id."""
    Xs, Ks = [], []
    for c in countries:
        want = None if ids_by_country is None else ids_by_country.get(c)
        if ids_by_country is not None and (want is None or not len(want)):
            continue
        X, K = load_features(split, c, bcfg, want)
        K = K.copy()
        K["country"] = c
        Xs.append(X)
        Ks.append(K)
    if not Xs:
        raise ValueError("nothing gathered")
    return np.vstack(Xs), pd.concat(Ks, ignore_index=True)


def sort_by_s1(X: np.ndarray, K: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    order = np.lexsort((K["cid"].to_numpy(), K["src"].to_numpy(), K["s1"].to_numpy()))
    return X[order], K.iloc[order].reset_index(drop=True)


def add_meta(X: np.ndarray, K: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    """Append the nine meta columns.  Rows must already be sorted by S1."""
    s1 = K["s1"].to_numpy()
    _, s1_codes = np.unique(s1, return_inverse=True)
    ckey = K["src"].to_numpy().astype(np.int64) * (1 << 32) + K["cid"].to_numpy().astype(np.int64)
    _, cand_codes = np.unique(ckey, return_inverse=True)
    M = build_meta_features(s1_codes.astype(np.int64), cand_codes.astype(np.int64),
                            K["src"].to_numpy().astype(np.int8), prob)
    return np.hstack([X, M])


# ------------------------------------------------------------------ decision
def predictions_to_dict(K: pd.DataFrame, keep: np.ndarray) -> Dict[str, List[str]]:
    sel = K.loc[keep]
    out: Dict[str, List[str]] = {}
    for a, b, c in zip(sel["s1"].to_numpy(), sel["src"].to_numpy(), sel["cid"].to_numpy()):
        out.setdefault(f"S1-{a}", []).append(f"S{b}-{c}")
    return out


def apply_threshold(K: pd.DataFrame, prob: np.ndarray, t: float) -> Dict[str, List[str]]:
    return predictions_to_dict(K, prob >= t)


def expected_f_masks(K: pd.DataFrame, prob: np.ndarray, T: float,
                     miss_probs: Sequence[float]) -> List[np.ndarray]:
    """One selection mask per ``miss_prob``, from a single pass over the entities.

    The Poisson-binomial ladders and the convolution behind ``E[F_k]`` do not
    depend on the miss probability — only the two weights that mix the complete
    and one-missing cases do — so a whole sweep costs what one value used to.
    See :func:`ber.decide.expected_f05_components`.
    """
    from .decide import decide_sweep
    s1 = K["s1"].to_numpy()
    p = temper(prob, T) if T != 1.0 else np.asarray(prob, dtype=np.float64)
    bounds = group_starts(s1)
    keeps = [np.zeros(len(p), dtype=bool) for _ in miss_probs]
    for a, b in zip(bounds[:-1], bounds[1:]):
        for keep, (chosen, _) in zip(keeps, decide_sweep(p[a:b], miss_probs)):
            if chosen:
                keep[a + np.fromiter(chosen, dtype=np.int64, count=len(chosen))] = True
    return keeps


def expected_f_mask(K: pd.DataFrame, prob: np.ndarray, T: float = 1.0,
                    miss_prob: float = 0.0) -> np.ndarray:
    """Boolean mask of the rows the expected-F_0.5 rule selects.

    A mask rather than a dict: at test scale the dict-of-lists form costs
    gigabytes of Python strings, while the mask is one byte per candidate.
    Rows must already be sorted by S1.
    """
    return expected_f_masks(K, prob, T, (miss_prob,))[0]


def decision_mask(K: pd.DataFrame, prob: np.ndarray, rule: Tuple) -> np.ndarray:
    """Boolean mask for either decision rule."""
    if rule[0] == "threshold":
        return np.asarray(prob) >= rule[1]
    return expected_f_mask(K, prob, T=rule[1],
                           miss_prob=rule[2] if len(rule) > 2 else 0.0)


def apply_expected_f(K: pd.DataFrame, prob: np.ndarray, T: float = 1.0,
                     miss_prob: float = 0.0) -> Dict[str, List[str]]:
    """Dict form of :func:`expected_f_mask`, for validation-sized inputs."""
    keep = expected_f_mask(K, prob, T=T, miss_prob=miss_prob)
    out: Dict[str, List[str]] = {f"S1-{i}": [] for i in K["s1"].to_numpy()}
    sel = K.loc[keep]
    for a, b, c in zip(sel["s1"].to_numpy(), sel["src"].to_numpy(), sel["cid"].to_numpy()):
        out[f"S1-{a}"].append(f"S{b}-{c}")
    return out


#: Temperatures and candidate-miss probabilities the decision search sweeps.
#:
#: Both grids were widened after the first full run, and for the same reason: it
#: chose ``T = 0.6`` for both countries, which was the *edge* of the old set, and
#: ``miss_prob`` was never swept at all — the rule always assumed the candidate
#: list was complete.  It is not: India's pruned full-cluster recall is 0.754, so
#: for a quarter of entities at least one true match never reached the matcher.
#: Telling the rule that raises the true-set size it budgets for, which makes one
#: more predicted candidate cheaper and the empty prediction worthless, and at
#: this operating point recall is worth about 4.6x precision per unit
#: (P = 0.974, R = 0.838 -> dlnF/dlnR = 0.82 against dlnF/dlnP = 0.18).
DECISION_TEMPS: Tuple[float, ...] = (0.35, 0.45, 0.55, 0.6, 0.7, 0.8, 1.0, 1.25, 1.6)
DECISION_MISS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def search_decision(K: pd.DataFrame, prob: np.ndarray, gold: Dict[str, Iterable[str]],
                    keys: Sequence[str],
                    thresholds: Sequence[float] = tuple(np.arange(0.30, 0.96, 0.025)),
                    temps: Sequence[float] = DECISION_TEMPS,
                    miss_probs: Sequence[float] = DECISION_MISS
                    ) -> Tuple[Tuple, float, Dict[Tuple, float]]:
    """Tuned global threshold vs the expected-F rule; returns the winner and all scores."""
    res: Dict[Tuple, float] = {}
    for t in thresholds:
        res[("threshold", round(float(t), 3))] = macro_f05(
            apply_threshold(K, prob, t), gold, keys)
    all_keys = [f"S1-{i}" for i in K["s1"].to_numpy()]
    for T in temps:
        for mp, keep in zip(miss_probs, expected_f_masks(K, prob, T, miss_probs)):
            res[("eum", T, mp)] = macro_f05(_pred_dict(K, keep, all_keys), gold, keys)
    best = max(res, key=res.get)
    return best, res[best], res


def _pred_dict(K: pd.DataFrame, keep: np.ndarray,
               all_keys: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in all_keys}
    sel = K.loc[keep]
    for a, b, c in zip(sel["s1"].to_numpy(), sel["src"].to_numpy(), sel["cid"].to_numpy()):
        out[f"S1-{a}"].append(f"S{b}-{c}")
    return out


def apply_decision(K: pd.DataFrame, prob: np.ndarray, rule: Tuple) -> Dict[str, List[str]]:
    if rule[0] == "threshold":
        return apply_threshold(K, prob, rule[1])
    return apply_expected_f(K, prob, T=rule[1], miss_prob=rule[2] if len(rule) > 2 else 0.0)
