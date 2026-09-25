import numpy as np
import pandas as pd
import pytest

from ber.preranker import PRERANK_FEATURES, matrix, prune
from ber.retrieval import RETRIEVAL_COLUMNS


def _df(s1, prob_col=None):
    n = len(s1)
    d = {"s1": np.asarray(s1, dtype=np.int32)}
    for c in PRERANK_FEATURES:
        d[c] = np.zeros(n, dtype=np.float32)
    return pd.DataFrame(d)


def test_prerank_features_exclude_their_own_output():
    assert "pre" in RETRIEVAL_COLUMNS
    assert "pre" not in PRERANK_FEATURES


def test_matrix_shape():
    df = _df([1, 1, 2])
    assert matrix(df).shape == (3, len(PRERANK_FEATURES))


def test_prune_keeps_min_even_below_eps():
    df = _df([1] * 5)
    p = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    keep = prune(df, p, max_keep=4, min_keep=3, eps=0.5)
    assert keep.sum() == 3


def test_prune_caps_at_max_even_above_eps():
    df = _df([1] * 10)
    p = np.full(10, 0.9)
    keep = prune(df, p, max_keep=4, min_keep=2, eps=0.5)
    assert keep.sum() == 4


def test_prune_is_per_s1():
    df = _df([1, 1, 1, 2, 2])
    p = np.array([0.9, 0.8, 0.1, 0.9, 0.05])
    keep = prune(df, p, max_keep=3, min_keep=1, eps=0.5)
    assert list(keep) == [True, True, False, True, False]


def test_prune_keeps_the_highest_scoring_rows():
    df = _df([7] * 6)
    p = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7])
    keep = prune(df, p, max_keep=3, min_keep=0, eps=0.0)
    assert set(np.flatnonzero(keep)) == {1, 3, 5}


def test_prune_handles_unsorted_input():
    df = _df([2, 1, 2, 1])
    p = np.array([0.9, 0.2, 0.1, 0.8])
    keep = prune(df, p, max_keep=1, min_keep=1, eps=0.0)
    assert list(keep) == [True, False, False, True]


def test_model_path_is_keyed_by_the_blocking_config():
    """A pre-ranker fitted to one retriever set must not be reused for another."""
    from ber.preranker import model_path
    from ber.retrieval import BlockingConfig

    a = model_path(BlockingConfig())
    b = model_path(BlockingConfig(k_per_retriever=10))
    c = model_path(BlockingConfig(query_budget=30_000))
    assert a != b != c and a != c
    assert all(str(p).endswith(".pkl") for p in (a, b, c))
    assert model_path() != a
