"""The feature driver must attach the right record text to every candidate row."""
import numpy as np
import pandas as pd
import pytest

from ber.featurize import Block, _rows_for, _lookup, compute_block
from ber.features import N_PAIR, I_RETRIEVAL, I_SET, PAIR_FEATURES
from ber.normalize import TokenRoles
from ber.retrieval import RETRIEVAL_COLUMNS


def test_lookup_and_rows_for_roundtrip():
    ids = np.array([50, 10, 30, 20], dtype=np.int32)
    sorted_ids, order = _lookup(ids)
    want = np.array([30, 50, 99], dtype=np.int32)
    rows = _rows_for(sorted_ids, order, want)
    assert ids[rows[0]] == 30
    assert ids[rows[1]] == 50
    assert rows[2] == -1               # 99 is absent


def test_rows_for_on_an_empty_want():
    ids = np.array([1, 2, 3], dtype=np.int32)
    s, o = _lookup(ids)
    assert len(_rows_for(s, o, np.zeros(0, dtype=np.int32))) == 0


@pytest.fixture
def roles():
    brands = ["apex", "beta", "gamma", "delta"]
    names = [f"{b} {o} digital llc" for b in brands for o in brands if b != o]
    return TokenRoles(min_share=0.01, bigram_min_share=0.01).fit(
        names, ["US"] * len(names), ["1 main st"] * len(names), min_df=1)


def test_compute_block_shapes_and_set_columns(roles):
    n = 4
    block = Block(
        country="US",
        q_name=["Apex Digital LLC"], q_addr=["1795 Brackendale Dr"],
        c_name=["Apex Digital", "Apex Digital Inc", "Beta Tools", "Apex Digital"],
        c_addr=["1795 Brackendale Drive", "1799 Brackendale Dr", "2 Other St",
                "001795 Brackendale Drive"],
        q_row=np.zeros(n, dtype=np.int32),
        c_row=np.arange(n, dtype=np.int32),
        src=np.array([2, 2, 3, 3], dtype=np.int8),
        retr=np.tile(np.linspace(0.1, 0.9, len(RETRIEVAL_COLUMNS)).astype(np.float32),
                     (n, 1)),
        bounds=np.array([0, n], dtype=np.int64),
    )
    X = compute_block(block, roles)
    assert X.shape == (n, N_PAIR)
    cols = dict(zip(PAIR_FEATURES, X.T))
    # exact number match beats the neighbouring-number candidate
    assert cols["num_rel"][0] == 1.0
    assert cols["num_rel"][1] == 4.0        # 1795 vs 1799 -> differ 3-20
    assert cols["num_rel"][3] == 1.0        # zero padding is still exact
    # the list-level columns are filled for the whole group
    assert (cols["x_n_cands"] == n).all()
    assert np.isfinite(cols["x_rrf_rank"]).all()
    # retrieval columns came straight from the candidate table
    assert np.allclose(X[:, I_RETRIEVAL:I_SET], block.retr)


def test_compute_block_support_is_cross_source(roles):
    """x_support_* compares a candidate with the best candidate of the OTHER source."""
    block = Block(
        country="US",
        q_name=["Apex Digital LLC"], q_addr=["1 Main St"],
        c_name=["Apex Digital", "Apex Digital"], c_addr=["1 Main St", "1 Main St"],
        q_row=np.zeros(2, dtype=np.int32), c_row=np.arange(2, dtype=np.int32),
        src=np.array([2, 3], dtype=np.int8),
        retr=np.zeros((2, len(RETRIEVAL_COLUMNS)), dtype=np.float32),
        bounds=np.array([0, 2], dtype=np.int64),
    )
    X = compute_block(block, roles)
    cols = dict(zip(PAIR_FEATURES, X.T))
    assert cols["x_support_name"][0] == pytest.approx(1.0)
    assert cols["x_support_addr"][0] == pytest.approx(1.0)


def test_compute_block_support_is_nan_without_the_other_source(roles):
    block = Block(
        country="US",
        q_name=["Apex Digital LLC"], q_addr=["1 Main St"],
        c_name=["Apex Digital"], c_addr=["1 Main St"],
        q_row=np.zeros(1, dtype=np.int32), c_row=np.zeros(1, dtype=np.int32),
        src=np.array([2], dtype=np.int8),
        retr=np.zeros((1, len(RETRIEVAL_COLUMNS)), dtype=np.float32),
        bounds=np.array([0, 1], dtype=np.int64),
    )
    X = compute_block(block, roles)
    cols = dict(zip(PAIR_FEATURES, X.T))
    assert np.isnan(cols["x_support_name"][0])
