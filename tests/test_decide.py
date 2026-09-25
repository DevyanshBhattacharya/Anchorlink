"""decide() must match exhaustive search over all 2**n subsets (doc, Module 5.1)."""
import itertools
import math

import numpy as np
import pytest

from ber.decide import (decide, expected_f05_topk, poisson_binomial, temper,
                        exclusivity, exclusivity_arrays, resolve_double_claims)
from ber.scoring import f05

BETA2 = 0.25


def brute_expected_f05(p, subset):
    """E[F_0.5] of predicting ``subset``, marginalising over all truth patterns."""
    n = len(p)
    total = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        prob = 1.0
        for pi, b in zip(p, bits):
            prob *= pi if b else (1.0 - pi)
        if prob == 0.0:
            continue
        true = {i for i, b in enumerate(bits) if b}
        tp = len(subset & true)
        total += prob * f05(tp, len(subset), len(true))
    return total


def brute_best(p):
    n = len(p)
    best_val, best_set = -1.0, None
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            v = brute_expected_f05(p, set(combo))
            if v > best_val + 1e-12:
                best_val, best_set = v, set(combo)
    return best_set, best_val


def test_poisson_binomial_matches_brute_force():
    rng = np.random.default_rng(0)
    p = rng.random(7)
    dist = poisson_binomial(p)
    brute = np.zeros(len(p) + 1)
    for bits in itertools.product((0, 1), repeat=len(p)):
        pr = 1.0
        for pi, b in zip(p, bits):
            pr *= pi if b else 1 - pi
        brute[sum(bits)] += pr
    assert np.allclose(dist, brute, atol=1e-12)


def test_expected_f_values_match_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(1, 8))
        p = rng.random(n)
        order, ef = expected_f05_topk(p)
        for k in range(n + 1):
            want = brute_expected_f05(p, set(order[:k].tolist()))
            assert math.isclose(ef[k], want, rel_tol=1e-9, abs_tol=1e-12)


def test_decide_matches_exhaustive_search():
    """300 random instances, exactly the check described in the doc."""
    rng = np.random.default_rng(20260925)
    for trial in range(300):
        n = int(rng.integers(1, 8))
        style = trial % 3
        if style == 0:
            p = rng.random(n)
        elif style == 1:
            p = rng.beta(0.4, 0.4, n)          # pushed to the extremes
        else:
            p = np.clip(rng.normal(0.5, 0.25, n), 0.001, 0.999)
        chosen, val = decide(p)
        best_set, best_val = brute_best(p)
        assert math.isclose(val, best_val, rel_tol=1e-9, abs_tol=1e-12), (p, chosen, best_set)


def test_doc_worked_examples():
    # a lone candidate at 0.40 -> empty (expected 0.60); at 0.55 -> match
    chosen, val = decide([0.40])
    assert chosen == set()
    assert math.isclose(val, 0.60, rel_tol=1e-9)
    chosen, _ = decide([0.55])
    assert chosen == {0}
    # [0.90, 0.60, 0.35, 0.20] -> only the first
    chosen, _ = decide([0.90, 0.60, 0.35, 0.20])
    assert chosen == {0}


def test_empty_candidate_list():
    chosen, val = decide([])
    assert chosen == set()
    assert val == 1.0


def test_miss_prob_makes_empty_less_attractive():
    """A blocking-lossy segment should stop over-predicting the empty set."""
    prev = None
    for m in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        _, ef = expected_f05_topk(np.array([0.40]), miss_prob=m)
        if prev is not None:
            assert ef[0] < prev            # empty keeps losing value
        prev = ef[0]
    assert int(np.argmax(expected_f05_topk(np.array([0.40]), miss_prob=0.0)[1])) == 0
    assert int(np.argmax(expected_f05_topk(np.array([0.40]), miss_prob=0.5)[1])) == 1


def test_temper_is_monotone_and_identity_at_one():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert np.allclose(temper(p, 1.0), p)
    for T in (0.5, 0.8, 1.5, 2.0):
        q = temper(p, T)
        assert np.all(np.diff(q) > 0)


def test_exclusivity_doc_example():
    out = exclusivity({"S1-a": {"S2-x": 0.90}, "S1-b": {"S2-x": 0.60}})
    assert math.isclose(out["S1-a"]["S2-x"], 0.78, abs_tol=0.005)
    assert math.isclose(out["S1-b"]["S2-x"], 0.13, abs_tol=0.005)


def test_exclusivity_uncontested_is_identity():
    out = exclusivity({"S1-a": {"S2-x": 0.9}})
    assert math.isclose(out["S1-a"]["S2-x"], 0.9, rel_tol=1e-9)


def test_exclusivity_arrays_matches_dict_version():
    p_by_s1 = {"a": {"x": 0.9, "y": 0.3}, "b": {"x": 0.6}, "c": {"y": 0.8, "z": 0.2}}
    s1_keys = sorted(p_by_s1)
    cand_keys = sorted({c for d in p_by_s1.values() for c in d})
    s1_code = {k: i for i, k in enumerate(s1_keys)}
    c_code = {k: i for i, k in enumerate(cand_keys)}
    si, ci, pp = [], [], []
    for s, d in p_by_s1.items():
        for c, v in d.items():
            si.append(s1_code[s]); ci.append(c_code[c]); pp.append(v)
    got = exclusivity_arrays(np.array(si), np.array(ci), np.array(pp))
    want_d = exclusivity(p_by_s1)
    for k, (s, c) in enumerate(zip(si, ci)):
        assert math.isclose(got[k], want_d[s1_keys[s]][cand_keys[c]], rel_tol=1e-9)


def test_resolve_double_claims():
    pred = {"a": {"x", "y"}, "b": {"x"}}
    probs = {"a": {"x": 0.7, "y": 0.9}, "b": {"x": 0.8}}
    out = resolve_double_claims(pred, probs)
    assert out["a"] == {"y"}
    assert out["b"] == {"x"}


def _naive_expected_f(p, beta2=0.25, miss=0.0):
    """The O(n**3) form the convolution version replaced, kept as the oracle."""
    p = np.clip(np.asarray(p, float), 0, 1)
    order = np.argsort(-p, kind="stable")
    q = p[order]
    n = len(q)
    ef = np.empty(n + 1)
    ef[0] = float(np.prod(1 - q)) * (1 - miss)
    for k in range(1, n + 1):
        A, B = poisson_binomial(q[:k]), poisson_binomial(q[k:])
        a = np.arange(k + 1)[:, None].astype(float)
        b = np.arange(n - k + 1)[None, :].astype(float)
        val = 1.25 * a * ((1 - miss) / (beta2 * (a + b) + k)
                          + miss / (beta2 * (a + b + 1) + k))
        ef[k] = (A[:, None] * B[None, :] * val).sum()
    return ef


def test_convolution_form_matches_the_naive_form_exactly():
    """The fast rule must be the same arithmetic, not an approximation."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(1, 40))
        p = rng.beta(0.4, 0.4, n)
        miss = float(rng.choice([0.0, 0.15, 0.4]))
        _, fast = expected_f05_topk(p, miss_prob=miss)
        worst = max(worst, float(np.max(np.abs(fast - _naive_expected_f(p, miss=miss)))))
    assert worst < 1e-12, worst


def test_prefix_and_suffix_ladders_match_direct_computation():
    from ber.decide import _prefix_poisson_binomials, _suffix_poisson_binomials
    rng = np.random.default_rng(5)
    q = rng.random(9)
    A = _prefix_poisson_binomials(q)
    B = _suffix_poisson_binomials(q)
    for k in range(len(q) + 1):
        assert np.allclose(A[k], poisson_binomial(q[:k]), atol=1e-14)
        assert np.allclose(B[k], poisson_binomial(q[k:]), atol=1e-14)
