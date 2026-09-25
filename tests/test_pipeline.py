"""Decision search, meta assembly and the pieces that glue stages together."""
import numpy as np
import pandas as pd
import pytest

from ber.pipeline import (apply_decision, apply_expected_f, apply_threshold,
                          predictions_to_dict, search_decision, sort_by_s1, sub_fold)
from ber.scoring import macro_f05


@pytest.fixture
def table():
    K = pd.DataFrame({
        "s1": np.array([1, 1, 1, 2, 2, 3], dtype=np.int32),
        "src": np.array([2, 2, 3, 2, 3, 3], dtype=np.int8),
        "cid": np.array([10, 11, 12, 20, 21, 30], dtype=np.int32),
    })
    p = np.array([0.95, 0.10, 0.80, 0.55, 0.05, 0.30])
    gold = {"S1-1": ["S2-10", "S3-12"], "S1-2": ["S2-20"], "S1-3": []}
    keys = ["S1-1", "S1-2", "S1-3"]
    return K, p, gold, keys


def test_threshold_rule(table):
    K, p, gold, keys = table
    pred = apply_threshold(K, p, 0.5)
    assert pred["S1-1"] == ["S2-10", "S3-12"]
    assert pred["S1-2"] == ["S2-20"]
    assert "S1-3" not in pred or pred["S1-3"] == []
    assert macro_f05(pred, gold, keys) == pytest.approx(1.0)


def test_expected_f_rule_is_per_entity(table):
    K, p, gold, keys = table
    pred = apply_expected_f(K, p)
    assert set(pred["S1-1"]) == {"S2-10", "S3-12"}
    assert pred["S1-3"] == []            # 0.30 alone is not worth emitting


def test_expected_f_emits_nothing_for_a_weak_lone_candidate():
    K = pd.DataFrame({"s1": np.array([9], np.int32), "src": np.array([2], np.int8),
                      "cid": np.array([1], np.int32)})
    assert apply_expected_f(K, np.array([0.40]))["S1-9"] == []
    assert apply_expected_f(K, np.array([0.55]))["S1-9"] == ["S2-1"]


def test_search_decision_finds_a_perfect_rule(table):
    K, p, gold, keys = table
    rule, score, tbl = search_decision(K, p, gold, keys)
    assert score == pytest.approx(1.0)
    assert rule[0] in ("threshold", "eum")
    assert len(tbl) > 5
    assert apply_decision(K, p, rule)


def test_search_decision_rules_are_reproducible(table):
    K, p, gold, keys = table
    a = search_decision(K, p, gold, keys)[2]
    b = search_decision(K, p, gold, keys)[2]
    assert a == b


def test_sort_by_s1_groups_rows(table):
    K, p, gold, keys = table
    shuffled = K.iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
    X = np.arange(len(shuffled) * 3, dtype=np.float32).reshape(len(shuffled), 3)
    Xs, Ks = sort_by_s1(X, shuffled)
    s1 = Ks["s1"].to_numpy()
    assert (np.diff(s1) >= 0).all()
    # rows stayed attached to their keys
    for row, cid in zip(Xs, Ks["cid"]):
        orig = np.flatnonzero(shuffled["cid"].to_numpy() == cid)[0]
        assert np.array_equal(row, X[orig])


def test_predictions_to_dict_prefixes_correctly(table):
    K, p, gold, keys = table
    out = predictions_to_dict(K, p > 0.5)
    assert out["S1-1"] == ["S2-10", "S3-12"]


def test_sub_fold_is_a_deterministic_partition():
    ids = np.arange(1000, dtype=np.int32)
    a = sub_fold(ids, 0)
    b = sub_fold(ids, 1)
    assert set(a.tolist()) | set(b.tolist()) == set(ids.tolist())
    assert not (set(a.tolist()) & set(b.tolist()))
    assert np.array_equal(a, sub_fold(ids, 0))


def test_expected_f_mask_matches_the_dict_form(table):
    from ber.pipeline import apply_expected_f, expected_f_mask
    K, p, gold, keys = table
    mask = expected_f_mask(K, p)
    as_dict = apply_expected_f(K, p)
    rebuilt = {}
    for a, b, c in zip(K.loc[mask, "s1"], K.loc[mask, "src"], K.loc[mask, "cid"]):
        rebuilt.setdefault(f"S1-{a}", []).append(f"S{b}-{c}")
    for k, v in as_dict.items():
        assert sorted(v) == sorted(rebuilt.get(k, []))


def test_decision_mask_handles_both_rules(table):
    from ber.pipeline import decision_mask
    K, p, gold, keys = table
    assert list(decision_mask(K, p, ("threshold", 0.5))) == list(p >= 0.5)
    m = decision_mask(K, p, ("eum", 1.0, 0.0))
    assert m.dtype == bool and len(m) == len(p)
