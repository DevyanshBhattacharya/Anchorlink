"""The sparse index must retrieve the obvious pairs and survive pruning."""
import numpy as np
import pytest

from ber.blocking import (IndexConfig, SparseIndex, char_wb_ngrams, make_view,
                          word_ngrams)


DOCS = [
    ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC"),
    ("Prime Money", "17560 Ellis Road, Tahlequah, OK"),
    ("Orelees Barber Shop", "1795 Westchester Dr, High Point, North Carolina"),
    ("B+ Retail Inc", "1712 Montebello Avenue, Phoenix, AZ"),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
    ("Ram Marketing Private Limited", "KH No 570/13, New Delhi, West Delhi"),
]


def _cfg(**kw):
    base = dict(min_df=1, max_df_abs=10 ** 9, max_df_frac=1.0)
    base.update(kw)
    return IndexConfig(**base)


def build(cfg):
    return SparseIndex(cfg).build([make_view(cfg, n, a) for n, a in DOCS])


def test_char_wb_is_word_bounded():
    assert char_wb_ngrams("ab cdef", 3) == [" ab", "ab ", " cd", "cde", "def", "ef "]
    assert char_wb_ngrams("a", 3) == [" a "]
    assert word_ngrams("a b c") == ["a", "b", "c"]
    assert word_ngrams("a b", 3) == ["a b"]      # shorter than n: keep the whole text
    assert word_ngrams("", 3) == []


def test_views():
    cfg = _cfg(view="name")
    assert make_view(cfg, "Orelee's Barbershop", "1795 X") == "orelee s barbershop"
    assert make_view(_cfg(view="addr"), "N", "1795 X") == "1795 x"
    with pytest.raises(ValueError):
        make_view(_cfg(view="nope"), "a", "b")


def test_retrieves_the_noisy_twin():
    idx = build(_cfg(kind="char+word", ngram=4))
    q = make_view(idx.cfg, *DOCS[0])
    I, S = idx.query([q], k=3)
    assert I[0, 0] == 0                 # itself
    assert 2 in I[0].tolist()           # the abbreviated twin


def test_skeleton_view_bridges_scripts():
    cfg = _cfg(kind="char", ngram=3, view="skeleton")
    idx = build(cfg)
    I, S = idx.query([make_view(cfg, *DOCS[4])], k=3)
    assert 5 in I[0].tolist()           # the Latin spelling of the same business


def test_pruning_keeps_the_top_hit():
    idx = build(_cfg(kind="char+word", ngram=4, query_budget=5, min_query_terms=2))
    q = make_view(idx.cfg, *DOCS[0])
    I_pruned, _ = idx.query([q], k=3, prune=True)
    I_full, _ = idx.query([q], k=3, prune=False)
    assert I_pruned[0, 0] == I_full[0, 0] == 0


def test_empty_query_yields_nothing():
    idx = build(_cfg())
    I, S = idx.query([""], k=3)
    assert (I[0] == -1).all()


def test_unfilled_slots_are_minus_one():
    idx = build(_cfg())
    I, S = idx.query([make_view(idx.cfg, *DOCS[1])], k=20)
    assert I.shape == (1, 20)
    assert (I[0] == -1).sum() > 0
    assert (S[0][I[0] == -1] == 0).all()


def test_df_cap_removes_ubiquitous_grams():
    cfg = IndexConfig(min_df=1, max_df_abs=3, max_df_frac=1.0, kind="word", ngram=1)
    docs = ["common a", "common b", "common c", "common d", "common e"]
    idx = SparseIndex(cfg).build(docs)
    assert "common" not in idx.vocab
    assert "a" in idx.vocab


def test_save_load_roundtrip(tmp_path):
    idx = build(_cfg(kind="char+word", ngram=4))
    p = tmp_path / "idx.pkl"
    idx.save(p)
    back = SparseIndex.load(p)
    q = make_view(idx.cfg, *DOCS[0])
    a, _ = idx.query([q], k=3)
    b, _ = back.query([q], k=3)
    assert a.tolist() == b.tolist()


def test_scores_are_cosines_in_range():
    idx = build(_cfg(kind="char+word", ngram=4))
    _, S = idx.query([make_view(idx.cfg, *DOCS[0])], k=6)
    assert S.max() <= 1.0 + 1e-5
    assert S.min() >= 0.0


def test_empty_documents_at_every_position():
    """Trailing empty rows used to send add.reduceat out of bounds."""
    cfg = IndexConfig(min_df=1, max_df_abs=10 ** 9, max_df_frac=1.0, kind="char+word", ngram=4)
    for docs in (["", "alpha widgets", ""],
                 ["", "", "alpha widgets"],
                 ["alpha widgets", "", ""],
                 ["", "", ""],
                 ["alpha widgets"]):
        idx = SparseIndex(cfg).build(docs)
        I, S = idx.query(["alpha widgets"], k=2)
        assert I.shape == (1, 2)
        if any(docs):
            assert I[0, 0] == docs.index("alpha widgets")


def test_row_norms_are_one_for_non_empty_docs():
    from ber.blocking import _row_reduce
    cfg = IndexConfig(min_df=1, max_df_abs=10 ** 9, max_df_frac=1.0, kind="word", ngram=1)
    idx = SparseIndex(cfg).build(["a b c", "", "d e"])
    D = idx.Dt.T.tocsr()
    norms = np.sqrt(_row_reduce(D.data * D.data, D.indptr, D.shape[0]))
    assert np.allclose(norms[[0, 2]], 1.0)
    assert norms[1] == 0.0


def test_openmp_health_check_fires_once_and_names_offenders(monkeypatch):
    import sys as _sys
    import ber.blocking as B
    monkeypatch.setattr(B, "_OMP_WARNED", False)
    monkeypatch.setitem(_sys.modules, "lightgbm", object())
    msg = B.check_openmp_health()
    assert msg and "lightgbm" in msg
    assert B.check_openmp_health() is None      # only once


def test_openmp_health_check_is_silent_when_clean(monkeypatch):
    import sys as _sys
    import ber.blocking as B
    monkeypatch.setattr(B, "_OMP_WARNED", False)
    for m in ("lightgbm", "sklearn"):
        monkeypatch.delitem(_sys.modules, m, raising=False)
    assert B.check_openmp_health() is None
