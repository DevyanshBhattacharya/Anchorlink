"""The matcher: a monotone-constrained LightGBM stack plus calibration.

Two stages, both LightGBM with binary log-loss (never LambdaRank — the decision
layer in Module 5 needs probabilities, not orderings):

* **M1** scores each (S1, candidate) pair from the pairwise, retrieval and
  candidate-list features.
* **M2 (meta)** adds the nine columns that only exist once M1 has scored the
  whole list: rank, gap to best, exclusivity-renormalised probability, how many
  S1 entities compete for this candidate, and the best score from the other
  source.

Monotone constraints ("more name similarity is never worse, more number conflict
is never better") remove non-monotone splits that fit noise in US/India and would
not transfer to France.  Stacking is out-of-fold by name family, so no S1 or its
matches appear in both a model's training and scoring fold.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

from . import config
from .features import META_FEATURES, PAIR_FEATURES, monotone_vector

if TYPE_CHECKING:            # pragma: no cover
    import lightgbm as lgb


def _lgb():
    """Import LightGBM on first use, never at module import.

    LightGBM ships its own OpenMP runtime.  Loading it into a process that later
    runs the sparse top-k product drops that product to a single thread and makes
    blocking about ten times slower (PROGRESS.md §7), so the retrieval stages must
    be able to import this package without pulling LightGBM in.
    """
    import lightgbm
    return lightgbm

BASE_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    monotone_constraints_method="advanced",
    verbose=-1,
    num_threads=0,
    seed=config.SEED,
    deterministic=True,
    force_row_wise=True,
)


@dataclass
class TrainedModel:
    booster: "lgb.Booster"
    feature_names: List[str]
    calibrator: object | None = None        # sklearn IsotonicRegression
    meta: Dict = field(default_factory=dict)

    def raw_predict(self, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        return self.booster.predict(X, num_iteration=num_iteration or
                                    self.booster.best_iteration or None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.raw_predict(X)
        if self.calibrator is not None:
            p = self.calibrator.predict(np.clip(p, 0.0, 1.0))
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self.booster.model_to_string(),
                         "feature_names": self.feature_names,
                         "calibrator": self.calibrator,
                         "meta": self.meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(_lgb().Booster(model_str=d["model"]), d["feature_names"],
                   d["calibrator"], d.get("meta", {}))


def train_lgb(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
              params: Dict | None = None, num_boost_round: int = 3000,
              early_stopping: int = 100, monotone: bool = True,
              verbose_eval: int = 0) -> TrainedModel:
    lgb = _lgb()
    p = dict(BASE_PARAMS)
    if params:
        p.update(params)
    if monotone:
        p["monotone_constraints"] = monotone_vector(feature_names)
    names = list(feature_names)
    dtrain = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False)
    valid, callbacks = [dtrain], []
    if X_val is not None and len(X_val):
        dval = lgb.Dataset(X_val, label=y_val, feature_name=names, reference=dtrain,
                           free_raw_data=False)
        valid = [dval]
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    if verbose_eval:
        callbacks.append(lgb.log_evaluation(verbose_eval))
    booster = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                        valid_sets=valid, callbacks=callbacks)
    return TrainedModel(booster, names, None,
                        {"params": {k: v for k, v in p.items() if k != "monotone_constraints"},
                         "best_iteration": booster.best_iteration,
                         "n_train": int(len(y)), "pos_rate": float(y.mean())})


def fit_calibrator(p: np.ndarray, y: np.ndarray):
    """Isotonic regression on a held-out fold.  Monotone, so rankings survive."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0),
            np.asarray(y, dtype=np.float64))
    return iso


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    p = np.clip(np.asarray(p, float), 0, 1)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Dict]:
    p = np.clip(np.asarray(p, float), 0, 1)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"[{edges[b]:.2f},{edges[b+1]:.2f})",
                         "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 4),
                         "frac_pos": round(float(y[m].mean()), 4)})
    return rows


def feature_importance(model: TrainedModel, top: int = 25) -> List[Tuple[str, float]]:
    gain = model.booster.feature_importance(importance_type="gain")
    pairs = sorted(zip(model.feature_names, gain), key=lambda t: -t[1])
    total = float(gain.sum()) or 1.0
    return [(n, round(100.0 * g / total, 2)) for n, g in pairs[:top]]
