"""The submission writer must stream, stay in file order and never quote."""
import numpy as np
import pytest

from ber import config, run


@pytest.fixture
def fake_test_source(tmp_path, monkeypatch):
    p = tmp_path / "test_source1.tsv"
    ids = ["S1-300", "S1-100", "S1-200", "S1-400"]      # deliberately unsorted
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in ids:
            f.write(f"{i}\tNA\t1 Main\tUS\n")
    monkeypatch.setitem(config.FILES, ("test", "S1"), p)
    return ids


def test_write_rows_follows_source_order_and_dedups(tmp_path, fake_test_source):
    s1 = np.array([100, 100, 100, 200, 300], dtype=np.int32)
    src = np.array([2, 3, 2, 2, 3], dtype=np.int8)
    cid = np.array([1, 2, 1, 9, 7], dtype=np.int32)      # (2,1) repeats for S1-100
    keep = np.array([True, True, True, False, True])
    out = tmp_path / "m.tsv"
    rows, non_empty, total = run._write_rows(
        out, ["source1_entity_id", "matched_entity_ids"], s1, src, cid, keep)
    raw = out.read_bytes()
    assert b"\r" not in raw and b'"' not in raw and b", " not in raw
    lines = raw.decode().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-300\tS3-7"        # source order, not sorted order
    assert lines[2] == "S1-100\tS2-1,S3-2"   # duplicate (2,1) collapsed
    assert lines[3] == "S1-200\t"            # keep=False -> empty
    assert lines[4] == "S1-400\t"            # absent from the scored table
    assert (rows, non_empty, total) == (4, 2, 3)


def test_write_rows_without_a_mask_emits_every_candidate(tmp_path, fake_test_source):
    s1 = np.array([100, 100], dtype=np.int32)
    src = np.array([2, 3], dtype=np.int8)
    cid = np.array([1, 2], dtype=np.int32)
    out = tmp_path / "c.tsv"
    rows, non_empty, total = run._write_rows(
        out, ["source1_entity_id", "candidate_entity_ids"], s1, src, cid, None)
    assert (rows, non_empty, total) == (4, 1, 2)
    assert out.read_text().split("\n")[2] == "S1-100\tS2-1,S3-2"


def test_write_rows_on_an_empty_scored_table(tmp_path, fake_test_source):
    empty_i = np.zeros(0, dtype=np.int32)
    out = tmp_path / "m.tsv"
    rows, non_empty, total = run._write_rows(
        out, ["source1_entity_id", "matched_entity_ids"],
        empty_i, np.zeros(0, np.int8), empty_i, np.zeros(0, bool))
    assert (rows, non_empty, total) == (4, 0, 0)
    assert out.read_text().strip().split("\n")[1] == "S1-300\t"
