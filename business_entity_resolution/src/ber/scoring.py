"""Macro F_0.5 exactly as the problem statement defines it.

Per Source-1 entity:

    F_0.5 = 1.25 * |P & T| / (0.25 * |T| + |P|)

with the singleton rule: an entity with no true matches scores 1.0 for an empty
prediction and 0.0 for any non-empty one, and an entity with true matches scores
0.0 for an empty prediction.  The reported number is the plain mean over *all*
Source-1 entities in the evaluation set, singletons included.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Set

import numpy as np

BETA2 = 0.25  # beta ** 2 for beta = 0.5


def f05(tp: int, n_pred: int, n_true: int) -> float:
    """Per-entity F_0.5 from the three counts, singleton rule included."""
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
    return (1.0 + BETA2) * tp / (BETA2 * n_true + n_pred)


def f05_sets(pred: Iterable[str], true: Iterable[str]) -> float:
    p, t = set(pred), set(true)
    return f05(len(p & t), len(p), len(t))


def macro_f05(pred: Mapping[str, Iterable[str]],
              gold: Mapping[str, Iterable[str]],
              keys: Sequence[str] | None = None) -> float:
    """Macro average over ``keys`` (default: every key in ``gold``).

    A key missing from ``pred`` counts as an empty prediction, which is what the
    scorer sees for an entity we failed to emit.
    """
    keys = list(gold.keys()) if keys is None else list(keys)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        t = gold.get(k, ())
        p = pred.get(k, ())
        ts = t if isinstance(t, (set, frozenset)) else set(t)
        ps = p if isinstance(p, (set, frozenset)) else set(p)
        total += f05(len(ps & ts), len(ps), len(ts))
    return total / len(keys)


def macro_f05_breakdown(pred: Mapping[str, Iterable[str]],
                        gold: Mapping[str, Iterable[str]],
                        keys: Sequence[str] | None = None) -> Dict[str, float]:
    """Macro F_0.5 plus the micro precision/recall behind it, for diagnostics."""
    keys = list(gold.keys()) if keys is None else list(keys)
    scores, tp_tot, p_tot, t_tot = [], 0, 0, 0
    n_singleton = n_singleton_hit = 0
    for k in keys:
        ts = set(gold.get(k, ()))
        ps = set(pred.get(k, ()))
        tp = len(ps & ts)
        scores.append(f05(tp, len(ps), len(ts)))
        tp_tot += tp
        p_tot += len(ps)
        t_tot += len(ts)
        if not ts:
            n_singleton += 1
            n_singleton_hit += int(not ps)
    micro_p = tp_tot / p_tot if p_tot else 0.0
    micro_r = tp_tot / t_tot if t_tot else 0.0
    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "n": len(keys),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "pred_ids": p_tot,
        "true_ids": t_tot,
        "tp": tp_tot,
        "n_singleton": n_singleton,
        "singleton_accuracy": n_singleton_hit / n_singleton if n_singleton else float("nan"),
        "avg_pred_size": p_tot / len(keys) if keys else 0.0,
    }


# ------------------------------------------------------------- blocking bound
def f05_ceiling(cand: Mapping[str, Iterable[str]],
                gold: Mapping[str, Iterable[str]],
                keys: Sequence[str] | None = None) -> Dict[str, float]:
    """The best macro F_0.5 any matcher could reach on this candidate set.

    For one entity with true set T and candidate set C the bound is
    ``1.25 |C & T| / (0.25 |T| + |C & T|)`` — predict exactly the recoverable
    true matches and nothing else.  Also returns pair recall, full-cluster
    recall, the average candidate-list size and the reduction ratio.
    """
    keys = list(gold.keys()) if keys is None else list(keys)
    bounds, n_cand = [], 0
    hit = tot = 0
    full = 0
    n_nonsingleton = 0
    for k in keys:
        ts = set(gold.get(k, ()))
        cs = set(cand.get(k, ()))
        n_cand += len(cs)
        inter = len(cs & ts)
        hit += inter
        tot += len(ts)
        bounds.append(f05(inter, inter, len(ts)))
        if ts:
            n_nonsingleton += 1
            full += int(inter == len(ts))
    return {
        "n": len(keys),
        "pair_recall": hit / tot if tot else float("nan"),
        "full_cluster_recall": full / n_nonsingleton if n_nonsingleton else float("nan"),
        "avg_candidates": n_cand / len(keys) if keys else 0.0,
        "f05_ceiling": float(np.mean(bounds)) if bounds else 0.0,
        "candidate_pairs": n_cand,
    }
