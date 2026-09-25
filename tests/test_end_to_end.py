"""Run the whole pipeline on a miniature three-country dataset.

The fixture's third country exists only in the test split, so this exercises the
unseen-country path (partition on the exact string, no hard-coded list) as well
as the writer, the validator and every stage in between.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "business_entity_resolution" / "src"
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    data = d / "dataset"
    subprocess.run([PY, str(ROOT / "tests" / "fixtures" / "make_synthetic.py"),
                    str(data), "--n", "1200"], check=True, capture_output=True)
    env = {**os.environ,
           "PYTHONPATH": str(SRC),
           "BER_ROOT": str(ROOT),
           "BER_DATA": str(data),
           "BER_WORK": str(d / "work"),
           "BER_OUTPUT": str(d / "output"),
           "HF_HUB_OFFLINE": "1"}
    res = subprocess.run(
        [PY, "-m", "ber.cli", "all", "--workers", "2",
         "--n-m1", "900", "--n-m2", "400", "--n-cal", "300", "--no-loco"],
        env=env, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        pytest.fail(f"pipeline failed:\nSTDOUT\n{res.stdout[-6000:]}\nSTDERR\n{res.stderr[-6000:]}")
    return d, env, res.stdout


def test_pipeline_runs(run_dir):
    d, env, out = run_dir
    assert "reweighted validation F0.5" in out


def test_both_tsvs_exist_and_are_lf_only(run_dir):
    d, env, _ = run_dir
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = d / "output" / name
        assert p.exists(), name
        raw = p.read_bytes()
        assert b"\r" not in raw
        assert b'"' not in raw
        assert b", " not in raw


def test_headers_are_exact(run_dir):
    d, _, _ = run_dir
    m = (d / "output" / "matching_results.tsv").read_text().split("\n", 1)[0]
    c = (d / "output" / "candidate_pairs.tsv").read_text().split("\n", 1)[0]
    assert m == "source1_entity_id\tmatched_entity_ids"
    assert c == "source1_entity_id\tcandidate_entity_ids"


def test_one_row_per_test_s1(run_dir):
    d, _, _ = run_dir
    s1 = [l.split("\t", 1)[0]
          for l in (d / "dataset" / "test" / "test_source1.tsv").read_text().split("\n")[1:] if l]
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        rows = [l.split("\t", 1)[0]
                for l in (d / "output" / name).read_text().split("\n")[1:] if l]
        assert rows == s1            # same set, same order, no duplicates


def test_unseen_country_gets_predictions(run_dir):
    """Carpathia appears only in test; it must still be matched, not just emitted."""
    d, _, _ = run_dir
    src1 = (d / "dataset" / "test" / "test_source1.tsv").read_text().split("\n")[1:]
    carpathia = {l.split("\t")[0] for l in src1 if l and l.split("\t")[3] == "Carpathia"}
    assert carpathia
    matched = 0
    for line in (d / "output" / "matching_results.tsv").read_text().split("\n")[1:]:
        if not line:
            continue
        s1, _, rest = line.partition("\t")
        if s1 in carpathia and rest.strip():
            matched += 1
    assert matched > 0.2 * len(carpathia), f"only {matched}/{len(carpathia)} Carpathia matched"


def test_matches_are_a_subset_of_candidates(run_dir):
    d, _, _ = run_dir

    def read(name):
        out = {}
        for line in (d / "output" / name).read_text().split("\n")[1:]:
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            out[s1] = set(x for x in rest.split(",") if x)
        return out
    m, c = read("matching_results.tsv"), read("candidate_pairs.tsv")
    for s1, ids in m.items():
        assert ids <= c[s1], s1


def test_official_validator_passes(run_dir):
    d, env, _ = run_dir
    res = subprocess.run(
        [PY, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
         "--matching", str(d / "output" / "matching_results.tsv"),
         "--candidate", str(d / "output" / "candidate_pairs.tsv"),
         "--test-dir", str(d / "dataset" / "test"), "--check-ids"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout


def test_report_has_the_expected_sections(run_dir):
    d, _, _ = run_dir
    rep = json.loads((d / "work" / "reports" / "phase0_report.json").read_text())
    assert set(rep) >= {"blocking", "stack", "validation", "headline", "output"}
    assert rep["headline"]["reweighted"] > 0.0
    for row in rep["blocking"]["pruned"]:
        assert 0.0 <= row["pair_recall"] <= 1.0
