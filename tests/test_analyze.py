"""Ablation plumbing and the error-pattern labeller."""
import numpy as np
import pandas as pd
import pytest

from ber.analyze import _pattern, decision_ablation, segment_report
from ber.scoring import f05


@pytest.mark.parametrize("qa,ca,want", [
    ("616 Kingsmere Ave", "627 Kingsmere Ave", "house numbers differ by 3-20 (same street, a few doors away)"),
    ("616 Kingsmere Ave", "617 Kingsmere Ave", "house numbers differ by 1-2 (same street, next door)"),
    ("1447 Harborview Blvd", "447 Harborview Blvd", "house number truncated (1447 -> 447)"),
    ("616 Kingsmere Ave", "99999 Other Rd", "house numbers unrelated"),
    ("616 Kingsmere Ave", "Kingsmere Ave", "no house number on one side"),
    ("616 Kingsmere Ave", "", "missing address on one side"),
])
def test_number_patterns(qa, ca, want):
    assert _pattern("Apex Digital", qa, "Apex Digital", ca) == want


def test_name_patterns_when_numbers_agree():
    assert _pattern("Apex Digital", "1 Main", "Apex Digital", "1 Main") == \
        "identical name, same number"
    assert _pattern("Apex Digital", "1 Main", "डायनामिक", "1 Main") == "native-script name"
    assert _pattern("8913 Textiles", "1 Main", "8913textiles.com", "1 Main") == \
        "domain-style name"
    assert _pattern("Apex Digital", "1 Main", "Zulu Widgets", "1 Main") == \
        "no shared name token"
    assert _pattern("Apex Digital", "1 Main", "Apex Digital LLC", "1 Main") == \
        "name differs by a legal form or one token"


def test_decision_ablation_reports_both_rules():
    K = pd.DataFrame({"s1": np.array([1, 1, 2], np.int32),
                      "src": np.array([2, 3, 2], np.int8),
                      "cid": np.array([10, 11, 20], np.int32)})
    gold = {"S1-1": ["S2-10"], "S1-2": []}
    keys = ["S1-1", "S1-2"]
    variants = {"raw": np.array([0.9, 0.2, 0.1]),
                "excl": np.array([0.95, 0.05, 0.05])}
    rows = decision_ablation(K, variants, gold, keys)
    assert {r["probabilities"] for r in rows} == {"raw", "excl"}
    for r in rows:
        assert 0.0 <= r["F0.5 @ tuned threshold"] <= 1.0
        assert 0.0 <= r["F0.5 @ expected-F"] <= 1.0
        assert r["F0.5"] >= max(r["F0.5 @ tuned threshold"], r["F0.5 @ expected-F"]) - 1e-9


def test_segment_report_splits_by_true_cluster_size():
    K = pd.DataFrame({"s1": np.array([1, 2, 2, 3, 3, 3], np.int32),
                      "src": np.array([2, 2, 3, 2, 3, 2], np.int8),
                      "cid": np.array([1, 2, 3, 4, 5, 6], np.int32)})
    gold = {"S1-1": [], "S1-2": ["S2-2", "S3-3"], "S1-3": ["S2-4", "S3-5", "S2-6"]}
    keys = ["S1-1", "S1-2", "S1-3"]
    y = np.array([0, 1, 1, 1, 1, 1])
    pred = np.array([False, True, True, True, True, True])
    rows = segment_report(K, np.ones(6) * 0.9, y, pred, gold, keys)
    by = {r["cluster size"]: r for r in rows}
    assert by["singleton"]["macro_f05"] == 1.0
    assert by["2-3"]["n"] == 2
    assert by["2-3"]["macro_f05"] == pytest.approx(1.0)
