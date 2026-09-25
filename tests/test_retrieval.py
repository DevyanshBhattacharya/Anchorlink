"""Fusion, capping and the cache fingerprint that stops a small run poisoning a big one."""
import numpy as np
import pytest

from ber.retrieval import (BlockingConfig, RETRIEVAL_COLUMNS, RETRIEVER_TAGS,
                           _query_stamp, _rank_within, fuse, MISSING_RANK)


def test_retrieval_columns_cover_every_retriever():
    for tag in RETRIEVER_TAGS:
        assert f"{tag}_score" in RETRIEVAL_COLUMNS
        assert f"{tag}_rank" in RETRIEVAL_COLUMNS
    assert RETRIEVAL_COLUMNS[-1] == "pre"


def test_specs_apply_the_per_view_df_cap():
    b = BlockingConfig()
    specs = dict(b.specs())
    # the skeleton view has a far larger cap: it has only ~11k possible 3-grams
    assert specs["sk"].max_df_abs > specs["na"].max_df_abs
    assert all(c.query_budget == b.query_budget for c in specs.values())


def test_df_scale_sweeps_every_view_at_once():
    a = dict(BlockingConfig(df_scale=1.0).specs())
    b = dict(BlockingConfig(df_scale=0.5).specs())
    for tag in RETRIEVER_TAGS:
        assert b[tag].max_df_abs == a[tag].max_df_abs // 2
        assert b[tag].max_df_frac == pytest.approx(a[tag].max_df_frac / 2)


def test_fuse_unions_and_records_every_retriever():
    nq = 2
    k = 3
    lists = []
    for li in range(len(RETRIEVER_TAGS)):
        I = np.full((nq, k), -1, dtype=np.int32)
        S = np.zeros((nq, k), dtype=np.float32)
        I[0, 0] = 5 + li            # each retriever finds a different doc for q0
        S[0, 0] = 0.9 - 0.1 * li
        I[1, 0] = 7                 # all agree on q1
        S[1, 0] = 0.5
        lists.append((I, S))
    out = fuse(nq, lists, None, max_per_source=0)
    assert set(out["q"].tolist()) == {0, 1}
    q1 = out["q"] == 1
    assert out["nret"][q1][0] == len(RETRIEVER_TAGS)
    q0 = out["q"] == 0
    assert q0.sum() == len(RETRIEVER_TAGS)          # four distinct docs
    assert out["nret"][q0].max() == 1
    # a retriever that did not see a doc leaves the sentinel rank
    assert (out[f"{RETRIEVER_TAGS[1]}_rank"][out["cid"] == 5] == MISSING_RANK).all()


def test_fuse_key_hits_do_not_change_rrf():
    nq = 1
    I = np.array([[3]], dtype=np.int32)
    S = np.array([[0.4]], dtype=np.float32)
    lists = [(I, S)] + [(np.full((1, 1), -1, np.int32), np.zeros((1, 1), np.float32))
                        for _ in RETRIEVER_TAGS[1:]]
    with_keys = fuse(nq, lists, [np.array([9], dtype=np.int32)], max_per_source=0)
    assert set(with_keys["cid"].tolist()) == {3, 9}
    key_row = with_keys["cid"] == 9
    assert with_keys["key_hit"][key_row][0] == 1
    assert with_keys["rrf"][key_row][0] == 0.0       # key hits contribute no RRF


def test_fuse_caps_per_source_keeping_the_best():
    nq = 1
    k = 5
    I = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
    S = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    lists = [(I, S)] + [(np.full((1, k), -1, np.int32), np.zeros((1, k), np.float32))
                        for _ in RETRIEVER_TAGS[1:]]
    out = fuse(nq, lists, None, max_per_source=2)
    assert len(out["cid"]) == 2
    assert set(out["cid"].tolist()) == {0, 1}        # best RRF wins


def test_fuse_on_nothing():
    nq = 3
    lists = [(np.full((nq, 2), -1, np.int32), np.zeros((nq, 2), np.float32))
             for _ in RETRIEVER_TAGS]
    out = fuse(nq, lists, None, max_per_source=5)
    assert len(out["q"]) == 0
    for tag in RETRIEVER_TAGS:
        assert len(out[f"{tag}_score"]) == 0


def test_rank_within_groups_on_unsorted_input():
    g = np.array([2, 1, 2, 1, 2])
    s = np.array([0.1, 0.9, 0.5, 0.2, 0.3])
    r = _rank_within(g, s)
    assert list(r) == [2, 0, 0, 1, 1]


def test_query_stamp_distinguishes_query_sets():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([3, 2, 1], dtype=np.int32)        # same set, different order
    c = np.array([1, 2, 4], dtype=np.int32)
    assert _query_stamp(a) == _query_stamp(b)
    assert _query_stamp(a) != _query_stamp(c)
    assert _query_stamp(None) == "ALL"
    assert _query_stamp(a) != "ALL"


def test_fingerprint_changes_with_every_knob():
    base = BlockingConfig()
    seen = {base.fingerprint()}
    for kw in (dict(k_per_retriever=25), dict(max_union_per_source=40),
               dict(df_scale=0.5), dict(query_budget=30_000),
               dict(use_keys=False), dict(prerank_max=30)):
        fp = BlockingConfig(**kw).fingerprint()
        assert fp not in seen, kw
        seen.add(fp)


def test_pruning_knobs_move_the_pruned_path_but_not_the_raw_one(tmp_path, monkeypatch):
    """The raw union does not depend on how it is later pruned.

    Re-running an hour of retrieval because ``prerank_eps`` changed is pure
    waste, and reusing a shard built with a *different* eps under the same name
    is a bug.  Both are decided here.
    """
    from ber import config, retrieval

    monkeypatch.setattr(config, "CAND_DIR", tmp_path)
    base = BlockingConfig()
    for kw in (dict(prerank_max=60), dict(prerank_min=8), dict(prerank_eps=0.0005)):
        other = BlockingConfig(**kw)
        assert (retrieval.cand_dir("train", "X", other, pruned=False)
                == retrieval.cand_dir("train", "X", base, pruned=False)), kw
        assert (retrieval.cand_dir("train", "X", other, pruned=True)
                != retrieval.cand_dir("train", "X", base, pruned=True)), kw
    # a retrieval knob moves both
    other = BlockingConfig(k_per_retriever=25)
    assert (retrieval.cand_dir("train", "X", other, pruned=False)
            != retrieval.cand_dir("train", "X", base, pruned=False))
    # and the key-block cap is part of what the raw union means
    assert (BlockingConfig(key_block_cap=150).retrieval_fingerprint()
            != base.retrieval_fingerprint())


def test_load_candidates_filters_per_row_group(tmp_path, monkeypatch):
    """The s1 filter must never materialise the whole table first."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ber import config, retrieval

    bcfg = BlockingConfig()
    monkeypatch.setattr(config, "CAND_DIR", tmp_path)
    d = retrieval.cand_dir("train", "Atlantis", bcfg, pruned=False)
    d.mkdir(parents=True)

    n = 5000
    rows = {"s1": np.repeat(np.arange(n // 10, dtype=np.int32), 10),
            "src": np.full(n, 2, dtype=np.int8),
            "cid": np.arange(n, dtype=np.int32)}
    for tag in RETRIEVER_TAGS:
        rows[f"{tag}_score"] = np.zeros(n, dtype=np.float32)
        rows[f"{tag}_rank"] = np.zeros(n, dtype=np.int16)
    rows["key_hit"] = np.zeros(n, dtype=np.int8)
    rows["rrf"] = np.zeros(n, dtype=np.float32)
    rows["nret"] = np.zeros(n, dtype=np.int8)
    pq.write_table(pa.table(rows, schema=retrieval.CAND_SCHEMA), d / "S2.parquet")

    want = np.array([3, 17, 499], dtype=np.int32)
    got = retrieval.load_candidates("train", "Atlantis", bcfg, s1_ids=want)
    assert set(got["s1"].tolist()) == {3, 17, 499}
    assert len(got) == 30
    # column subset still works, and an empty filter yields an empty frame
    got2 = retrieval.load_candidates("train", "Atlantis", bcfg,
                                     columns=["s1", "src", "cid"], s1_ids=want)
    assert list(got2.columns) == ["s1", "src", "cid"]
    empty = retrieval.load_candidates("train", "Atlantis", bcfg,
                                      s1_ids=np.array([99999], dtype=np.int32))
    assert len(empty) == 0
    # unfiltered read returns everything
    assert len(retrieval.load_candidates("train", "Atlantis", bcfg)) == n
