"""The scorer must match the problem statement exactly, singleton rule included."""
import math

import pytest

from ber.scoring import f05, f05_sets, macro_f05, macro_f05_breakdown, f05_ceiling


def test_readme_example():
    # predicted [S2-00047, S2-00193, S3-00812], true [S2-00047, S3-00812]
    got = f05_sets(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert math.isclose(got, 0.7142857142857143, rel_tol=1e-12)
    assert round(got, 3) == 0.714


@pytest.mark.parametrize("tp,npred,ntrue,want", [
    (0, 0, 0, 1.0),      # correct singleton
    (0, 1, 0, 0.0),      # false merge on a singleton
    (0, 3, 0, 0.0),
    (0, 0, 1, 0.0),      # missed everything
    (1, 1, 1, 1.0),      # perfect single
    (1, 1, 2, 1.25 / 1.5),
    (2, 2, 2, 1.0),
    (2, 3, 2, 2.5 / 3.5),
    (3, 3, 4, 1.25 * 3 / (1.0 + 3)),
    (4, 5, 4, 1.25 * 4 / (1.0 + 5)),
    (1, 1, 4, 0.625),
    (0, 1, 1, 0.0),
])
def test_table(tp, npred, ntrue, want):
    assert math.isclose(f05(tp, npred, ntrue), want, rel_tol=1e-12)


def test_doc_table_values():
    """The 'what you predict' table from the approach doc."""
    assert round(f05(1, 1, 2), 2) == 0.83       # 1 of 2 correct
    assert round(f05(2, 3, 2), 2) == 0.71       # all correct + 1 wrong, 2 true
    assert round(f05(1, 2, 1), 2) == 0.56       # all correct + 1 wrong, 1 true
    assert round(f05(3, 3, 4), 2) == 0.94       # 3 of 4
    assert round(f05(4, 5, 4), 2) == 0.83       # 4 of 4 + 1 wrong
    assert math.isclose(f05(1, 1, 4), 0.625)    # 1 of 4 (doc rounds to 0.63)


def test_macro_average_includes_singletons():
    gold = {"a": ["S2-1"], "b": [], "c": ["S2-2", "S3-3"]}
    pred = {"a": ["S2-1"], "b": [], "c": ["S2-2"]}
    expected = (1.0 + 1.0 + f05(1, 1, 2)) / 3
    assert math.isclose(macro_f05(pred, gold), expected, rel_tol=1e-12)


def test_missing_key_is_empty_prediction():
    gold = {"a": ["S2-1"], "b": []}
    assert math.isclose(macro_f05({}, gold), 0.5, rel_tol=1e-12)


def test_breakdown():
    gold = {"a": ["S2-1", "S2-2"], "b": []}
    pred = {"a": ["S2-1", "S2-9"], "b": ["S3-1"]}
    b = macro_f05_breakdown(pred, gold)
    assert b["n"] == 2
    assert b["tp"] == 1
    assert b["pred_ids"] == 3
    assert b["true_ids"] == 2
    assert b["n_singleton"] == 1
    assert b["singleton_accuracy"] == 0.0


def test_ceiling():
    gold = {"a": ["S2-1", "S2-2"], "b": ["S3-1"], "c": []}
    cand = {"a": ["S2-1", "S2-9"], "b": ["S3-1", "S3-2"], "c": ["S2-5"]}
    c = f05_ceiling(cand, gold)
    assert math.isclose(c["pair_recall"], 2 / 3, rel_tol=1e-12)
    assert math.isclose(c["full_cluster_recall"], 0.5, rel_tol=1e-12)
    assert math.isclose(c["avg_candidates"], 5 / 3, rel_tol=1e-12)
    # a: 1 of 2 recoverable -> 0.833 ; b: perfect -> 1.0 ; c: singleton -> 1.0
    assert math.isclose(c["f05_ceiling"], (f05(1, 1, 2) + 1.0 + 1.0) / 3, rel_tol=1e-12)
