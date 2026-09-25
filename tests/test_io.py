"""Loaders must not eat 'NA'-like names; writers must emit LF and no quoting."""
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ber import io_utils


NASTY = [
    ("S1-1", "NA", "1 Main St, Springfield, IL", "US"),
    ("S1-2", "NaN", "2 Main St", "US"),
    ("S1-3", "None", "3 Main St", "US"),
    ("S1-4", "NULL", "4 Main St", "US"),
    ("S1-5", 'He said "hi" Inc', "5 Main, St #7", "US"),
    ("S1-6", "n/a Services", "", "India"),
    ("S1-7", "-1.#IND Ltd", "7 Rue", "France"),
    ("S1-8", "inf", "8 Road", "US"),
]


@pytest.fixture
def nasty_tsv(tmp_path):
    p = tmp_path / "src.tsv"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for row in NASTY:
            f.write("\t".join(row) + "\n")
    return p


def test_read_tsv_keeps_na_like_names_as_strings(nasty_tsv):
    df = io_utils.read_tsv(nasty_tsv)
    assert list(df["business_name"]) == [r[1] for r in NASTY]
    assert df["business_name"].map(type).eq(str).all()
    assert df.loc[5, "business_address"] == ""       # empty stays empty, not NaN


def test_quote_none_keeps_embedded_quotes(nasty_tsv):
    df = io_utils.read_tsv(nasty_tsv)
    assert df.loc[4, "business_name"] == 'He said "hi" Inc'


def test_writer_is_lf_only_and_unquoted(tmp_path):
    path = tmp_path / "matching_results.tsv"
    mapping = {"S1-1": ["S2-1", "S3-2"], "S1-2": [], "S1-3": ["S3-9", "S3-9", "S1-4", "S2-7"]}
    rows, ne, tot = io_utils.write_matching(path, mapping, ["S1-1", "S1-2", "S1-3", "S1-4"])
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert b'"' not in raw
    assert b", " not in raw
    assert rows == 4 and ne == 2 and tot == 4
    lines = raw.decode().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-1,S3-2"
    assert lines[2] == "S1-2\t"                       # singleton: id then tab
    assert lines[3] == "S1-3\tS3-9,S2-7"              # dedup, S1 self-match dropped
    assert lines[4] == "S1-4\t"                       # missing key -> empty row
    assert lines[5] == ""                             # trailing newline


def test_writer_preserves_first_seen_order(tmp_path):
    path = tmp_path / "c.tsv"
    io_utils.write_candidates(path, {"S1-1": ["S3-3", "S2-1", "S3-3", "S2-2"]}, ["S1-1"])
    assert path.read_text().split("\n")[1] == "S1-1\tS3-3,S2-1,S2-2"


def test_read_id_list_roundtrip(tmp_path):
    path = tmp_path / "m.tsv"
    mapping = {"S1-1": ["S2-1", "S3-2"], "S1-2": []}
    io_utils.write_matching(path, mapping, ["S1-1", "S1-2"])
    back = io_utils.read_id_list(path)
    assert back == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}


def test_ground_truth_reader(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text("source1_entity_id\tmatched_entity_ids\n"
                 "S1-1\tS2-1,S3-2\n"
                 "S1-2\t\n", encoding="utf-8")
    gt = io_utils.read_ground_truth(p)
    assert gt == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}


def test_output_passes_official_validator(tmp_path):
    """End-to-end: build a miniature test set and run utils/validate_submission.py."""
    root = Path(__file__).resolve().parents[1]
    validator = root / "student_resource" / "utils" / "validate_submission.py"
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (test_dir / "test_source1.tsv").write_text(
        hdr + "S1-1\tNA\t1 Main\tUS\nS1-2\tB\t2 Main\tFrance\nS1-3\tC\t3 Main\tIndia\n",
        encoding="utf-8")
    (test_dir / "test_source2.tsv").write_text(
        hdr + "S2-1\tA\t1 Main\tUS\nS2-2\tB\t2 Main\tFrance\n", encoding="utf-8")
    (test_dir / "test_source3.tsv").write_text(
        hdr + "S3-1\tA\t1 Main\tUS\n", encoding="utf-8")

    out = tmp_path / "output"
    all_s1 = ["S1-1", "S1-2", "S1-3"]
    cands = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"], "S1-3": []}
    matches = {"S1-1": ["S2-1", "S3-1"], "S1-2": [], "S1-3": []}
    io_utils.write_candidates(out / "candidate_pairs.tsv", cands, all_s1)
    io_utils.write_matching(out / "matching_results.tsv", matches, all_s1)

    res = subprocess.run(
        [sys.executable, str(validator),
         "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"),
         "--test-dir", str(test_dir), "--check-ids"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout


def test_csv_writer_crlf_trap_is_real(tmp_path):
    """Documents why we never use csv.writer: it would glue \\r onto the last ID."""
    p = tmp_path / "bad.tsv"
    with open(p, "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        w.writerow(["S1-1", "S2-1,S3-2"])
    assert b"\r" in p.read_bytes()
