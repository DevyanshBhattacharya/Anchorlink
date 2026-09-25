import numpy as np
import pandas as pd
import pytest

from ber.expand import expand_two_hop


def _cand():
    return pd.DataFrame({
        "s1": np.array([1, 1, 2], dtype=np.int32),
        "src": np.array([2, 3, 2], dtype=np.int8),
        "cid": np.array([10, 20, 11], dtype=np.int32),
        "rrf": np.array([0.9, 0.5, 0.8], dtype=np.float32),
        "na_rank": np.array([0, 1, 0], dtype=np.int16),
    })


def test_two_hop_adds_the_other_source_neighbour():
    cand = _cand()
    # S2 row 0 (id 10) has S3 neighbour at pool row 1 -> id 21
    neighbours = {2: (np.array([[1, -1, -1]], dtype=np.int32), np.zeros((1, 3), np.float32)),
                  3: (np.array([[0, -1, -1]], dtype=np.int32), np.zeros((1, 3), np.float32))}
    pool_pos = {2: {10: 0, 11: 0}, 3: {20: 0}}
    pool_ids = {2: np.array([10, 11], dtype=np.int32),
                3: np.array([20, 21], dtype=np.int32)}
    out = expand_two_hop(cand, neighbours, pool_pos, pool_ids, top_seeds=1, per_seed=3)
    assert list(out.columns) == list(cand.columns)
    got = set(zip(out["s1"], out["src"], out["cid"]))
    assert (1, 3, 21) in got


def test_two_hop_never_repeats_an_existing_candidate():
    cand = _cand()
    neighbours = {2: (np.array([[0, -1, -1]], dtype=np.int32), np.zeros((1, 3), np.float32)),
                  3: (np.array([[0, -1, -1]], dtype=np.int32), np.zeros((1, 3), np.float32))}
    pool_pos = {2: {10: 0, 11: 0}, 3: {20: 0}}
    pool_ids = {2: np.array([10, 11], dtype=np.int32), 3: np.array([20], dtype=np.int32)}
    out = expand_two_hop(cand, neighbours, pool_pos, pool_ids, top_seeds=2, per_seed=3)
    existing = set(zip(cand["s1"], cand["src"], cand["cid"]))
    assert not (set(zip(out["s1"], out["src"], out["cid"])) & existing)


def test_two_hop_on_an_empty_table():
    cand = _cand().iloc[:0]
    out = expand_two_hop(cand, {}, {}, {})
    assert out.empty
