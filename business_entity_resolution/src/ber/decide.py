"""Module 5 — the per-entity decision layer.

The leaderboard score is a plain mean of independent per-entity scores, so
maximising expected F_0.5 separately for each Source-1 entity is exactly optimal.
Under independent marginals the optimal prediction is always a top-k prefix of
the candidates sorted by probability, so only n+1 sets need scoring instead of
2**n.

Three knobs sit in front of that rule:

* ``exclusivity`` — no S2/S3 record belongs to two S1 entities (0 violations in
  7,638,365 training pairs), so competing claims are renormalised in closed form.
* a temperature ``T`` on the final logit, the only free knob of the layer.
* ``miss_prob`` — the chance that blocking dropped a true match, which makes the
  empty prediction less attractive than the formula alone implies.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

BETA2 = 0.25


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """P(k successes) for independent Bernoulli(p_i), length len(p)+1."""
    dist = np.zeros(len(p) + 1, dtype=np.float64)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1.0 - pi) + dist[:-1] * pi
        dist[0] *= 1.0 - pi
    return dist


def _prefix_poisson_binomials(q: np.ndarray) -> List[np.ndarray]:
    """``A[k]`` = the count distribution of the first ``k`` candidates.

    Built incrementally — each step convolves the previous distribution with one
    Bernoulli — so the whole ladder costs O(n^2) instead of O(n^2) per level.
    """
    out = [np.array([1.0])]
    cur = np.array([1.0])
    for pi in q:
        nxt = np.zeros(len(cur) + 1)
        nxt[:-1] += cur * (1.0 - pi)
        nxt[1:] += cur * pi
        cur = nxt
        out.append(cur)
    return out


def _suffix_poisson_binomials(q: np.ndarray) -> List[np.ndarray]:
    """``B[k]`` = the count distribution of the candidates from ``k`` onwards."""
    n = len(q)
    out: List[np.ndarray] = [None] * (n + 1)      # type: ignore[list-item]
    cur = np.array([1.0])
    out[n] = cur
    for k in range(n - 1, -1, -1):
        pi = q[k]
        nxt = np.zeros(len(cur) + 1)
        nxt[:-1] += cur * (1.0 - pi)
        nxt[1:] += cur * pi
        cur = nxt
        out[k] = cur
    return out


def expected_f05_topk(p: np.ndarray, beta2: float = BETA2,
                      miss_prob: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(order, ef)`` — the descending order and E[F] for every prefix.

    ``ef[k]`` is the expected per-entity F_0.5 of predicting the top-k
    candidates.  ``miss_prob`` is the probability that at least one true match
    never entered the candidate list: it lowers the value of every prediction
    that would have been complete, so a blocking-lossy segment stops
    over-predicting the empty set.

    The naive form sums over (a, b) for every k, which is O(n^3).  Substituting
    ``s = a + b`` turns the inner double sum into a single convolution of
    ``a * A_k`` with ``B_k``, because the denominator depends only on ``s``:

        E[F_k] = (1+b2) * sum_s conv(a*A_k, B_k)[s] / (b2*s + k)

    Identical arithmetic, ~8x less of it.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    order = np.argsort(-p, kind="stable")
    q = p[order]
    n = len(q)
    ef = np.empty(n + 1, dtype=np.float64)
    if n == 0:
        ef[0] = 1.0 - miss_prob
        return order, ef

    A = _prefix_poisson_binomials(q)
    B = _suffix_poisson_binomials(q)
    ef[0] = float(np.prod(1.0 - q)) * (1.0 - miss_prob)
    for k in range(1, n + 1):
        Ak, Bk = A[k], B[k]
        num = np.convolve(np.arange(len(Ak), dtype=np.float64) * Ak, Bk)
        s = np.arange(len(num), dtype=np.float64)
        val = (1.0 + beta2) * ((1.0 - miss_prob) / (beta2 * s + k)
                               + miss_prob / (beta2 * (s + 1.0) + k))
        ef[k] = float(num @ val)
    return order, ef


def decide(p: Sequence[float], beta2: float = BETA2,
           miss_prob: float = 0.0) -> Tuple[Set[int], float]:
    """Indices of the expected-F_0.5-optimal subset, and its expected score."""
    arr = np.asarray(p, dtype=np.float64)
    order, ef = expected_f05_topk(arr, beta2=beta2, miss_prob=miss_prob)
    k = int(np.argmax(ef))
    return set(order[:k].tolist()), float(ef[k])


def temper(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature on the logit: T>1 flattens, T<1 sharpens.  Monotone."""
    if T == 1.0:
        return np.asarray(p, dtype=np.float64)
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return 1.0 / (1.0 + ((1.0 - q) / q) ** (1.0 / T))


# ------------------------------------------------------------- exclusivity
def exclusivity(p_by_s1: Mapping[str, Mapping[str, float]],
                eps: float = 1e-9) -> Dict[str, Dict[str, float]]:
    """Renormalise competing claims on the same S2/S3 record.

    With odds o = p/(1-p) and an at-most-one-owner prior,
    ``p'_{q,d} = o_{q,d} / (1 + sum_r o_{r,d})``.
    """
    odds: Dict[str, Dict[str, float]] = {}
    for s1, cands in p_by_s1.items():
        for c, pc in cands.items():
            odds.setdefault(c, {})[s1] = pc / max(1.0 - pc, eps)
    out: Dict[str, Dict[str, float]] = {}
    for s1, cands in p_by_s1.items():
        row = {}
        for c in cands:
            o = odds[c]
            row[c] = o[s1] / (1.0 + sum(o.values()))
        out[s1] = row
    return out


def exclusivity_arrays(s1_idx: np.ndarray, cand_idx: np.ndarray,
                       p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Vectorised ``exclusivity`` over a flat pair table.

    ``s1_idx`` and ``cand_idx`` are integer codes; ``p`` the probabilities.
    Returns the corrected probabilities in the same order.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0 - 1e-12)
    odds = p / np.maximum(1.0 - p, eps)
    n_cand = int(cand_idx.max()) + 1 if len(cand_idx) else 0
    tot = np.bincount(cand_idx, weights=odds, minlength=n_cand)
    return odds / (1.0 + tot[cand_idx])


def resolve_double_claims(pred: Mapping[str, Set[str]],
                          p_by_s1: Mapping[str, Mapping[str, float]]
                          ) -> Dict[str, Set[str]]:
    """Keep each S2/S3 id for the single S1 that claims it most strongly."""
    owner: Dict[str, Tuple[str, float]] = {}
    for s1, ids in pred.items():
        probs = p_by_s1.get(s1, {})
        for c in ids:
            pc = probs.get(c, 0.0)
            cur = owner.get(c)
            if cur is None or pc > cur[1] or (pc == cur[1] and s1 < cur[0]):
                owner[c] = (s1, pc)
    out: Dict[str, Set[str]] = {s1: set() for s1 in pred}
    for c, (s1, _) in owner.items():
        out[s1].add(c)
    return out
