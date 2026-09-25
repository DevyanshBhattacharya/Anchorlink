"""The contrastive objective and the batch sampler, checked exactly as the doc does."""
import numpy as np
import pytest

from ber.dense import (collision_batches, dense_topk, filter_mined_negatives,
                       serialise, supcon_loss)

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


def test_serialise_has_a_fixed_field_order():
    s = serialise("Apex Digital LLC", "1795 Westchester Dr", "US")
    assert s.startswith("name: ")
    assert " | nums: " in s and " | addr: " in s and " | country: US" in s
    assert "1795" in s.split("nums:")[1].split("|")[0]


def test_serialise_adds_a_romanised_echo_for_native_script():
    s = serialise("डायनामिक हॉस्पिटैलिटी", "12 MG Road", "India")
    assert "डायनामिक" in s
    assert "daynamik" in s.lower()


def test_serialise_handles_missing_fields():
    s = serialise("X", "", "France")
    assert "nums: -" in s and "addr: -" in s


def test_supcon_drops_from_random_to_clustered():
    """The doc's synthetic check: ~10.9 on random embeddings, ~1.1 on clustered."""
    g = torch.Generator().manual_seed(0)
    n_clusters, per = 60, 4
    cids = torch.arange(n_clusters).repeat_interleave(per)
    rand = torch.nn.functional.normalize(
        torch.randn(n_clusters * per, 32, generator=g), dim=1)
    loss_random = float(supcon_loss(rand, cids))

    centres = torch.nn.functional.normalize(torch.randn(n_clusters, 32, generator=g), dim=1)
    clustered = torch.nn.functional.normalize(
        centres.repeat_interleave(per, 0) + 0.01 * torch.randn(n_clusters * per, 32, generator=g),
        dim=1)
    loss_clustered = float(supcon_loss(clustered, cids))
    assert loss_random > 4 * loss_clustered
    assert loss_clustered < 2.0


def test_supcon_allows_many_positives():
    """One 4-member cluster, all embeddings identical: the loss is its floor.

    With B rows all in one cluster and all similarities equal, every anchor's
    log-prob is -log(B-1), so the loss is exactly log(B-1).  InfoNCE would have
    treated the three siblings as negatives and produced a much larger value.
    """
    import math
    same = supcon_loss(torch.nn.functional.normalize(torch.ones(4, 8), dim=1),
                       torch.zeros(4, dtype=torch.long))
    assert float(same) == pytest.approx(math.log(3), abs=1e-5)


def test_supcon_ignores_singletons():
    """Singletons act as negatives only; a batch of them has no gradient signal."""
    emb = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    cids = torch.arange(4)           # every row is its own cluster
    out = supcon_loss(emb, cids)
    assert float(out) == 0.0         # and finite: the -inf diagonal must not leak
    assert np.isfinite(float(out))


def test_collision_batches_cover_every_record_exactly_once():
    rng = np.random.default_rng(0)
    clusters, country, collisions = {}, {}, {}
    rid = 0
    for cid in range(1000):
        size = int(rng.integers(2, 7))
        clusters[cid] = list(range(rid, rid + size))
        rid += size
        country[cid] = "A" if cid % 2 else "B"
    for cid in clusters:
        collisions[cid] = [c for c in rng.integers(0, 1000, 6).tolist()
                           if country[c] == country[cid]]
    seen = []
    for batch in collision_batches(clusters, collisions, country, clusters_per_batch=64):
        ctry = {country[cid] for _, cid in batch}
        assert len(ctry) == 1              # never mixes countries
        seen.extend(r for r, _ in batch)
    assert len(seen) == rid
    assert len(set(seen)) == rid           # exactly once


def test_collision_batches_are_deterministic():
    clusters = {i: [i * 2, i * 2 + 1] for i in range(50)}
    country = {i: "A" for i in range(50)}
    collisions = {i: [(i + 1) % 50] for i in range(50)}
    a = [b for b in collision_batches(clusters, collisions, country, 8, seed=3)]
    b = [b for b in collision_batches(clusters, collisions, country, 8, seed=3)]
    assert a == b


def test_positive_aware_filtering():
    keep = filter_mined_negatives(np.array([0.5, 0.9, 0.7]), np.array([0.8, 0.8, 0.8]))
    assert list(keep) == [True, False, True]


def test_dense_topk_matches_brute_force():
    rng = np.random.default_rng(1)
    pool = rng.normal(size=(500, 16)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    q = pool[[3, 17, 400]] + 0.01 * rng.normal(size=(3, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx, sims = dense_topk(q, pool, k=5)
    assert idx.shape == (3, 5)
    assert list(idx[:, 0]) == [3, 17, 400]
    brute = q @ pool.T
    assert np.allclose(np.sort(sims, axis=1)[:, ::-1],
                       np.sort(brute, axis=1)[:, ::-1][:, :5], atol=1e-5)


def test_dense_topk_on_an_empty_pool():
    idx, sims = dense_topk(np.zeros((2, 4), np.float32), np.zeros((0, 4), np.float32), k=5)
    assert idx.shape == (2, 0)
