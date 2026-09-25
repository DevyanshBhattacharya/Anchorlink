"""Fit the matcher stack and measure it.

``fit_stack`` trains M1 on folds 1–3, M2 on fold 4a with M1's out-of-sample
scores, and isotonic calibration on fold 4b with M2's out-of-sample scores.
``score_rows`` runs the whole chain, applies the exclusivity correction and
returns calibrated probabilities.  Nothing here ever sees fold 0.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .decide import exclusivity_arrays
from .features import META_FEATURES, PAIR_FEATURES, build_meta_features
from .labels import GroundTruth
from .model import (TrainedModel, expected_calibration_error, feature_importance,
                    fit_calibrator, reliability_table, train_lgb)   # lazy LightGBM
from .pipeline import STAGE1_FEATURES, STAGE2_FEATURES, add_meta, gather, sort_by_s1

EARLY_STOP_FRACTION = 0.15


@dataclass
class Stack:
    m1: TrainedModel
    m2: TrainedModel
    calibrator: object | None = None
    info: Dict = field(default_factory=dict)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.m1.save(d / "m1.pkl")
        m2 = TrainedModel(self.m2.booster, self.m2.feature_names, self.calibrator,
                          self.m2.meta)
        m2.save(d / "m2.pkl")
        (d / "info.json").write_text(json.dumps(self.info, indent=1, default=str))

    @classmethod
    def load(cls, d: Path) -> "Stack":
        m1 = TrainedModel.load(d / "m1.pkl")
        m2 = TrainedModel.load(d / "m2.pkl")
        info = json.loads((d / "info.json").read_text()) if (d / "info.json").exists() else {}
        return cls(m1, m2, m2.calibrator, info)


def _holdout_by_s1(K: pd.DataFrame, frac: float, seed: int = config.SEED) -> np.ndarray:
    """Boolean mask selecting whole S1 groups for early stopping."""
    s1 = K["s1"].to_numpy()
    uniq = np.unique(s1)
    rng = np.random.default_rng(seed)
    pick = rng.random(len(uniq)) < frac
    chosen = uniq[pick]
    return np.isin(s1, chosen)


def fit_stack(split: str, countries: Sequence[str], bcfg, sel,
              gt: GroundTruth, verbose: bool = True) -> Stack:
    t0 = time.time()
    info: Dict = {"countries": list(countries)}

    # ---- M1 -------------------------------------------------------------
    X1, K1 = gather(split, countries, bcfg, sel.m1)
    X1, K1 = sort_by_s1(X1, K1)
    y1 = gt.label(K1["s1"].to_numpy(), K1["src"].to_numpy(), K1["cid"].to_numpy())
    hold = _holdout_by_s1(K1, EARLY_STOP_FRACTION)
    m1 = train_lgb(X1[~hold], y1[~hold], STAGE1_FEATURES,
                   X1[hold], y1[hold], num_boost_round=3000, early_stopping=100)
    info["m1"] = {"rows": int(len(y1)), "pos_rate": float(y1.mean()),
                  "best_iteration": m1.booster.best_iteration,
                  "top_features": feature_importance(m1, 15)}
    if verbose:
        print(f"[train] M1: {len(y1):,} rows, pos={y1.mean():.3f}, "
              f"iters={m1.booster.best_iteration}, {time.time()-t0:.0f}s", flush=True)
    del X1, K1, y1, hold
    gc.collect()

    # ---- M2 (meta) ------------------------------------------------------
    X2, K2 = gather(split, countries, bcfg, sel.m2)
    X2, K2 = sort_by_s1(X2, K2)
    p1 = m1.raw_predict(X2)
    X2m = add_meta(X2, K2, p1)
    y2 = gt.label(K2["s1"].to_numpy(), K2["src"].to_numpy(), K2["cid"].to_numpy())
    hold = _holdout_by_s1(K2, EARLY_STOP_FRACTION, seed=config.SEED + 7)
    m2 = train_lgb(X2m[~hold], y2[~hold], STAGE2_FEATURES,
                   X2m[hold], y2[hold], num_boost_round=3000, early_stopping=100)
    info["m2"] = {"rows": int(len(y2)), "pos_rate": float(y2.mean()),
                  "best_iteration": m2.booster.best_iteration,
                  "top_features": feature_importance(m2, 15)}
    if verbose:
        print(f"[train] M2: {len(y2):,} rows, iters={m2.booster.best_iteration}, "
              f"{time.time()-t0:.0f}s", flush=True)
    del X2, X2m, K2, y2, p1, hold
    gc.collect()

    # ---- calibration ----------------------------------------------------
    Xc, Kc = gather(split, countries, bcfg, sel.cal)
    Xc, Kc = sort_by_s1(Xc, Kc)
    pc = m2.raw_predict(add_meta(Xc, Kc, m1.raw_predict(Xc)))
    yc = gt.label(Kc["s1"].to_numpy(), Kc["src"].to_numpy(), Kc["cid"].to_numpy())
    cal = fit_calibrator(pc, yc)
    info["calibration"] = {
        "rows": int(len(yc)),
        "ece_raw": expected_calibration_error(pc, yc),
        "ece_calibrated": expected_calibration_error(np.clip(cal.predict(pc), 0, 1), yc),
        "reliability_raw": reliability_table(pc, yc),
    }
    if verbose:
        print(f"[train] calibration: ECE {info['calibration']['ece_raw']:.4f} -> "
              f"{info['calibration']['ece_calibrated']:.4f}", flush=True)
    del Xc, Kc, pc, yc
    gc.collect()
    info["seconds"] = round(time.time() - t0, 1)
    return Stack(m1, m2, cal, info)


def score_rows(stack: Stack, X: np.ndarray, K: pd.DataFrame,
               calibrate: bool = True, exclusive: bool = True
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Run M1 -> meta -> M2 -> calibration -> exclusivity.

    Returns ``(p_calibrated, p_exclusive)``; rows must already be sorted by S1.
    """
    p1 = stack.m1.raw_predict(X)
    p2 = stack.m2.raw_predict(add_meta(X, K, p1))
    p = np.clip(stack.calibrator.predict(np.clip(p2, 0, 1)), 1e-6, 1 - 1e-6) \
        if (calibrate and stack.calibrator is not None) else np.clip(p2, 1e-6, 1 - 1e-6)
    if not exclusive:
        return p, p
    s1 = K["s1"].to_numpy()
    _, s1_codes = np.unique(s1, return_inverse=True)
    ckey = K["src"].to_numpy().astype(np.int64) * (1 << 32) + K["cid"].to_numpy().astype(np.int64)
    _, c_codes = np.unique(ckey, return_inverse=True)
    pe = exclusivity_arrays(s1_codes.astype(np.int64), c_codes.astype(np.int64), p)
    return p, np.clip(pe, 1e-6, 1 - 1e-6)
