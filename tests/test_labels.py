import numpy as np
import pytest

from ber.labels import GroundTruth, pack


@pytest.fixture
def gt(tmp_path):
    p = tmp_path / "gt.tsv"
    p.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-100\tS2-1,S3-2\n"
        "S1-200\t\n"
        "S1-999998822\tS3-999999995\n", encoding="utf-8")
    return GroundTruth(p)


def test_pack_is_injective_and_fits_int64():
    s1 = np.array([999998822, 1, 999998822], dtype=np.int32)
    src = np.array([2, 3, 3], dtype=np.int8)
    cid = np.array([999999995, 1, 999999995], dtype=np.int32)
    k = pack(s1, src, cid)
    assert k.dtype == np.int64
    assert (k >= 0).all()
    assert len(np.unique(k)) == 3


def test_label(gt):
    s1 = np.array([100, 100, 100, 200, 999998822], dtype=np.int32)
    src = np.array([2, 3, 2, 2, 3], dtype=np.int8)
    cid = np.array([1, 2, 99, 1, 999999995], dtype=np.int32)
    assert list(gt.label(s1, src, cid)) == [1, 1, 0, 0, 1]


def test_source_matters(gt):
    """S2-1 is a match for S1-100; S3-1 is not."""
    assert gt.label(np.array([100]), np.array([2]), np.array([1]))[0] == 1
    assert gt.label(np.array([100]), np.array([3]), np.array([1]))[0] == 0


def test_cluster_size(gt):
    s = gt.cluster_size(np.array([100, 200, 999998822, 12345], dtype=np.int32))
    assert list(s) == [2, 0, 1, 0]
