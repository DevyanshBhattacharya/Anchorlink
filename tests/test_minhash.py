"""Banded MinHash must retrieve by similarity alone, never by rank."""
import numpy as np
import pytest

from ber.keys import KeyBlocks
from ber.minhash import MinHashConfig, MinHashIndex, _gather_bucket, permutations


def _pool(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "road", "street", "kochi",
             "columbus", "nc", "oh", "digital", "traders", "apex", "zeta"]
    return [" ".join(rng.choice(words, 6)) + f" {i}" for i in range(n)]


TARGET = "apex digital solutions 1795 westchester drive high point nc"
CORRUPT = "apex digitl solutons 1795 westchester dr high point nc"


def test_banding_threshold_matches_the_formula():
    cfg = MinHashConfig(bands=20, rows=3)
    assert cfg.n_perm == 60
    assert cfg.threshold == pytest.approx((1 / 20) ** (1 / 3))
    # more rows per band is a stricter retriever
    assert MinHashConfig(bands=20, rows=4).threshold > cfg.threshold


def test_permutations_are_deterministic():
    a1, b1 = permutations(MinHashConfig())
    a2, b2 = permutations(MinHashConfig())
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    a3, _ = permutations(MinHashConfig(seed=7))
    assert not np.array_equal(a1, a3)


def test_signature_agreement_estimates_jaccard():
    """The whole construction rests on P(slot agrees) = Jaccard."""
    cfg = MinHashConfig(bands=40, rows=4)          # 160 slots, for a tight estimate
    idx = MinHashIndex(cfg)
    sig = idx.signatures([TARGET, CORRUPT])
    agree = float((sig[0] == sig[1]).mean())
    a = set(cfg.analyse(TARGET))
    b = set(cfg.analyse(CORRUPT))
    true_j = len(a & b) / len(a | b)
    assert agree == pytest.approx(true_j, abs=0.12), (agree, true_j)


def test_a_corrupted_twin_is_retrieved_ahead_of_the_pool():
    """The point of the retriever: it is not a top-k contest.

    The twin does not have to out-rank three thousand distractors — it only has
    to collide — and the number of bands that collided ranks it above whatever
    incidental agreement the rest of the pool produces.
    """
    pool = _pool()
    pool[1234] = TARGET
    pool[2500] = CORRUPT
    # a generous bucket cap, so capping does not decide this test (it has its own)
    idx = MinHashIndex(MinHashConfig(max_bucket=500)).build(pool)
    I, S = idx.query([TARGET], k=10)
    hits = [int(x) for x in I[0] if x >= 0]
    assert hits[0] == 1234, "the exact record first"
    assert S[0][0] == pytest.approx(1.0), "an identical record agrees on every band"
    assert hits[1] == 2500, f"the twin second, got {hits}"
    assert 0.0 < S[0][1] < 1.0
    # and clear of anything else that happened to collide
    assert S[0][1] > S[0][2]


def test_rank_free_retrieval_survives_a_crowd_of_lookalikes():
    """A true match buried behind hundreds of near-duplicates is still found.

    This is the failure mode the four cosine views share: they are top-k, so the
    match at rank 200 is invisible.  Here the crowd does not compete.
    """
    crowd = ["apex digital solutions of somewhere else entirely"] * 400
    pool = crowd + _pool(600, seed=3) + [CORRUPT]
    idx = MinHashIndex(MinHashConfig()).build(pool)
    I, _ = idx.query([TARGET], k=30)
    assert (len(pool) - 1) in [int(x) for x in I[0] if x >= 0]


def test_query_grams_absent_from_the_pool_still_count():
    """Signatures are hashed, not looked up in a vocabulary, so an unseen gram
    lowers the estimated similarity instead of being silently dropped."""
    idx = MinHashIndex(MinHashConfig()).build([TARGET])
    sig_pool = idx.signatures([TARGET], grow=False)
    sig_novel = idx.signatures(["zzzz qqqq wwww vvvv"], grow=False)
    assert not np.array_equal(sig_pool, sig_novel)
    # and a query sharing nothing with the pool retrieves nothing
    I, _ = idx.query(["zzzz qqqq wwww vvvv"], k=5)
    assert [int(x) for x in I[0] if x >= 0] == []


def test_buckets_are_capped():
    cfg = MinHashConfig(max_bucket=5)
    idx = MinHashIndex(cfg).build(["the same text every time"] * 50)
    for kb in idx.bands:
        assert (np.diff(kb.starts) <= 5).all()


def test_empty_and_short_records_do_not_crash():
    idx = MinHashIndex(MinHashConfig()).build(["", "a", TARGET, ""])
    I, S = idx.query(["", TARGET], k=3)
    assert I.shape == (2, 3)
    assert 2 in [int(x) for x in I[1] if x >= 0]


def test_gather_bucket_matches_the_per_query_lookup():
    """The vectorised many-query gather must agree with KeyBlocks.lookup exactly.

    Twenty bands over twenty thousand queries is 400k Python-level lookups, so
    the gather is done with arithmetic; that arithmetic is what this pins.
    """
    keys = np.array([10, 10, 10, 20, 30, 30], dtype=np.int64)
    rows = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    kb = KeyBlocks.from_arrays(keys, rows, max_block=10)
    q = np.array([30, 999, 10, 20], dtype=np.int64)
    qi, got = _gather_bucket(kb, q)
    for i, key in enumerate(q):
        expect = sorted(kb.lookup([int(key)]).tolist())
        assert sorted(got[qi == i].tolist()) == expect, key


def test_save_and_load_round_trip(tmp_path):
    pool = _pool(500)
    pool[7] = TARGET
    idx = MinHashIndex(MinHashConfig()).build(pool)
    p = tmp_path / "lsh.pkl"
    idx.save(p)
    back = MinHashIndex.load(p)
    assert back.cfg == idx.cfg and back.n_docs == idx.n_docs
    a, _ = idx.query([TARGET], k=5)
    b, _ = back.query([TARGET], k=5)
    assert np.array_equal(a, b)
