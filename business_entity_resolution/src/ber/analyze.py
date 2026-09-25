"""Ablations and error analysis, run after the fact from the cached artefacts.

Everything here reads ``work/`` — feature shards, the trained stack, the
candidate tables — so the analysis can be rerun and extended without repeating
the expensive stages.
"""
from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .data import load_partition
from .decide import exclusivity_arrays
from .featurize import load_features
from .features import PAIR_FEATURES, number_relation
from .io_utils import read_ground_truth
from .labels import GroundTruth
from .normalize import has_native_script, house_numbers, is_domain_name, tokens
from .pipeline import (STAGE2_FEATURES, add_meta, apply_decision, apply_expected_f,
                       apply_threshold, search_decision, sort_by_s1)
from .scoring import macro_f05, macro_f05_breakdown
from .train import Stack


def _codes(K: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    _, s1_codes = np.unique(K["s1"].to_numpy(), return_inverse=True)
    ckey = K["src"].to_numpy().astype(np.int64) * (1 << 32) + K["cid"].to_numpy().astype(np.int64)
    _, c_codes = np.unique(ckey, return_inverse=True)
    return s1_codes.astype(np.int64), c_codes.astype(np.int64)


def score_variants(stack: Stack, X: np.ndarray, K: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Probabilities at each point of the chain, for the decision-layer ablation."""
    p1 = stack.m1.raw_predict(X)
    p2 = stack.m2.raw_predict(add_meta(X, K, p1))
    cal = (np.clip(stack.calibrator.predict(np.clip(p2, 0, 1)), 1e-6, 1 - 1e-6)
           if stack.calibrator is not None else np.clip(p2, 1e-6, 1 - 1e-6))
    s1c, cc = _codes(K)
    return {
        "m1_only": np.clip(p1, 1e-6, 1 - 1e-6),
        "m1_only_excl": np.clip(exclusivity_arrays(s1c, cc, np.clip(p1, 1e-6, 1 - 1e-6)),
                                1e-6, 1 - 1e-6),
        "m2_raw": np.clip(p2, 1e-6, 1 - 1e-6),
        "m2_calibrated": cal,
        "m2_calibrated_excl": np.clip(exclusivity_arrays(s1c, cc, cal), 1e-6, 1 - 1e-6),
        "m2_raw_excl": np.clip(exclusivity_arrays(s1c, cc, np.clip(p2, 1e-6, 1 - 1e-6)),
                               1e-6, 1 - 1e-6),
    }


def decision_ablation(K: pd.DataFrame, variants: Dict[str, np.ndarray],
                      gold: Dict[str, List[str]], keys: Sequence[str],
                      max_entities: int = 30_000,
                      seed: int = 20260925) -> List[Dict]:
    """Threshold vs expected-F, and the effect of calibration and exclusivity.

    This table is a *comparison* between six probability variants, not the
    headline number, so it runs on a subsample of whole S1 groups: six variants
    times thirty-odd candidate rules over the full fold would be an hour of
    expected-F evaluations to re-derive numbers the headline already reports.
    """
    if max_entities and len(keys) > max_entities:
        rng = np.random.default_rng(seed)
        keys = sorted(np.asarray(keys)[rng.permutation(len(keys))[:max_entities]].tolist())
        want = np.fromiter((int(k[3:]) for k in keys), dtype=np.int32, count=len(keys))
        m = np.isin(K["s1"].to_numpy(), want)
        K = K.loc[m].reset_index(drop=True)
        variants = {n: np.asarray(v)[m] for n, v in variants.items()}

    thresholds = tuple(np.round(np.arange(0.30, 0.96, 0.05), 3))
    rows = []
    for name, p in variants.items():
        rule, best, table = search_decision(K, p, gold, keys, thresholds=thresholds)
        thr = {k: v for k, v in table.items() if k[0] == "threshold"}
        eum = {k: v for k, v in table.items() if k[0] == "eum"}
        best_thr = max(thr, key=thr.get)
        best_eum = max(eum, key=eum.get)
        rows.append({
            "probabilities": name,
            "best_threshold": round(best_thr[1], 3),
            "F0.5 @ tuned threshold": round(thr[best_thr], 5),
            "F0.5 @ expected-F": round(eum[best_eum], 5),
            "expected-F temperature": best_eum[1],
            "best rule": str(rule),
            "F0.5": round(best, 5),
        })
    return rows


# ------------------------------------------------------------------ errors
def _pattern(qn: str, qa: str, cn: str, ca: str) -> str:
    """A short label for why a pair is hard, from the doc's noise taxonomy."""
    h1, h2 = house_numbers(qa), house_numbers(ca)
    rel = number_relation(h1, h2)
    if not qa.strip() or not ca.strip():
        return "missing address on one side"
    if rel == 1:
        pass
    elif rel == 2:
        return "house number truncated (1447 -> 447)"
    elif rel == 3:
        return "house numbers differ by 1-2 (same street, next door)"
    elif rel == 4:
        return "house numbers differ by 3-20 (same street, a few doors away)"
    elif rel == 5:
        return "house numbers unrelated"
    elif rel == 0:
        return "no house number on one side"
    if has_native_script(cn) or has_native_script(qn):
        return "native-script name"
    if is_domain_name(cn) or is_domain_name(qn):
        return "domain-style name"
    tq, tc = set(tokens(qn)), set(tokens(cn))
    if tq == tc:
        return "identical name, same number"
    if not (tq & tc):
        return "no shared name token"
    if len(tq ^ tc) <= 2:
        return "name differs by a legal form or one token"
    return "name partly overlapping"


def error_analysis(K: pd.DataFrame, prob: np.ndarray, y: np.ndarray,
                   pred_mask: np.ndarray, split: str, country: str,
                   n_examples: int = 5) -> Dict[str, object]:
    """Top false-positive and false-negative patterns, with examples."""
    s1p = load_partition(split, "S1", country)
    q = {int(i): (n, a) for i, n, a in zip(s1p.ids, s1p.names, s1p.addrs)}
    c: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for src in (2, 3):
        p = load_partition(split, f"S{src}", country)
        for i, n, a in zip(p.ids, p.names, p.addrs):
            c[(src, int(i))] = (n, a)
        del p
        gc.collect()
    del s1p
    gc.collect()

    s1a = K["s1"].to_numpy(); srca = K["src"].to_numpy(); cida = K["cid"].to_numpy()
    out: Dict[str, object] = {}
    for label, sel, order_desc in (("false_positives", pred_mask & (y == 0), True),
                                   ("false_negatives", (~pred_mask) & (y == 1), False)):
        idx = np.flatnonzero(sel)
        counts: Counter = Counter()
        for i in idx:
            qn, qa = q.get(int(s1a[i]), ("", ""))
            cn, ca = c.get((int(srca[i]), int(cida[i])), ("", ""))
            counts[_pattern(qn, qa, cn, ca)] += 1
        total = max(sum(counts.values()), 1)
        ranked = idx[np.argsort(-prob[idx] if order_desc else prob[idx])]
        examples = []
        for i in ranked[:n_examples]:
            qn, qa = q.get(int(s1a[i]), ("", ""))
            cn, ca = c.get((int(srca[i]), int(cida[i])), ("", ""))
            examples.append({"s1": f"S1-{int(s1a[i])}", "cand": f"S{int(srca[i])}-{int(cida[i])}",
                             "p": round(float(prob[i]), 4), "pattern": _pattern(qn, qa, cn, ca),
                             "s1_name": qn, "s1_addr": qa, "cand_name": cn, "cand_addr": ca})
        out[label] = {
            "n": int(len(idx)),
            "patterns": [{"pattern": k, "n": v, "share": round(v / total, 4)}
                         for k, v in counts.most_common(8)],
            "examples": examples,
        }
    return out


def segment_report(K: pd.DataFrame, prob: np.ndarray, y: np.ndarray,
                   pred_mask: np.ndarray, gold: Dict[str, List[str]],
                   keys: Sequence[str]) -> List[Dict]:
    """Macro F_0.5 split by true cluster size — where the score is won and lost."""
    sizes = {k: len(set(gold.get(k, ()))) for k in keys}
    pred: Dict[str, List[str]] = {}
    sel = K.loc[pred_mask]
    for a, b, d in zip(sel["s1"].to_numpy(), sel["src"].to_numpy(), sel["cid"].to_numpy()):
        pred.setdefault(f"S1-{a}", []).append(f"S{b}-{d}")
    rows = []
    for lo, hi, name in ((0, 0, "singleton"), (1, 1, "1 match"), (2, 3, "2-3"),
                         (4, 5, "4-5"), (6, 99, "6+")):
        ks = [k for k in keys if lo <= sizes.get(k, 0) <= hi]
        if not ks:
            continue
        b = macro_f05_breakdown(pred, gold, ks)
        rows.append({"cluster size": name, "n": b["n"],
                     "macro_f05": round(b["macro_f05"], 4),
                     "micro_precision": round(b["micro_precision"], 4),
                     "micro_recall": round(b["micro_recall"], 4)})
    return rows
