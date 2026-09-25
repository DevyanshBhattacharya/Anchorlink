"""Folds must be family-disjoint, balanced and deterministic."""
import numpy as np
import pandas as pd
import pytest

from ber import splits
from ber.normalize import TokenRoles


@pytest.fixture
def fam():
    rows = []
    for i in range(600):
        c = "US" if i % 2 else "India"
        rows.append({"id": np.int32(1000 + i), "country": c,
                     "family": f"fam{i % 120}", "fold": 0})
    df = pd.DataFrame(rows)
    df["fold"] = np.fromiter((splits.zlib.crc32(f.encode()) % splits.N_FOLDS
                              for f in df["family"]), dtype=np.int8, count=len(df))
    return df


def test_family_never_spans_two_folds(fam):
    per = fam.groupby("family")["fold"].nunique()
    assert (per == 1).all()


def test_folds_are_deterministic(fam):
    a = splits.fold_ids(fam, 1)
    b = splits.fold_ids(fam, 1)
    assert np.array_equal(a, b)


def test_fold_and_invert_partition_the_data(fam):
    a = set(splits.fold_ids(fam, 2).tolist())
    b = set(splits.fold_ids(fam, 2, invert=True).tolist())
    assert not (a & b)
    assert len(a | b) == len(fam)


def test_country_filter(fam):
    ids = splits.fold_ids(fam, 0, country="US")
    sub = fam[fam["id"].isin(ids)]
    assert set(sub["country"]) <= {"US"}


def test_sample_by_family_keeps_families_whole(fam):
    mask = (fam["country"].to_numpy() == "US")
    ids = splits.sample_by_family(fam, mask, 60)
    picked = fam[fam["id"].isin(ids)]
    chosen = set(picked["family"])
    all_us = fam[mask]
    for f in chosen:
        # every US member of a chosen family is in the sample
        assert set(all_us[all_us["family"] == f]["id"]) <= set(ids.tolist())


def test_sample_is_deterministic(fam):
    mask = np.ones(len(fam), dtype=bool)
    assert np.array_equal(splits.sample_by_family(fam, mask, 100),
                          splits.sample_by_family(fam, mask, 100))


def test_reweighted_score_matches_the_formula():
    mix = {"India": 0.47, "US": 0.38, "LOCO": 0.15}
    got = splits.reweighted_score({"India": 0.8, "US": 0.9}, loco=0.7, mix=mix)
    assert got == pytest.approx(0.47 * 0.8 + 0.38 * 0.9 + 0.15 * 0.7)


def test_default_mix_matches_the_measured_test_split():
    """The documented constant is what the derivation produces on the real files."""
    from ber import config
    assert config.TEST_MIX["India"] == pytest.approx(0.4675, abs=5e-4)
    assert config.TEST_MIX["US"] == pytest.approx(0.3827, abs=5e-4)
    assert config.TEST_MIX["LOCO"] == pytest.approx(0.1498, abs=5e-4)
    assert sum(config.TEST_MIX.values()) == pytest.approx(1.0, abs=1e-3)


def test_reweighted_score_renormalises_when_a_country_is_missing():
    mix = {"India": 0.47, "US": 0.38, "LOCO": 0.15}
    got = splits.reweighted_score({"India": 0.8}, loco=float("nan"), mix=mix)
    assert got == pytest.approx(0.8)


def test_loco_pairs(fam):
    assert set(splits.loco_pairs(fam)) == {("India", "US"), ("US", "India")}


def test_family_key_unions_lookalikes():
    roles = TokenRoles(min_share=0.01).fit(["alpha beta llc"] * 50, ["US"] * 50)
    a = splits._family_key("Dynamic Hospitality Private Limited", "India", None)
    b = splits._family_key("डायनामिक हॉस्पिटैलिटी प्राइवेट लिमिटेड", "India", None)
    assert a == b
    # word order does not create a new family
    c = splits._family_key("Hospitality Dynamic Private Limited", "India", None)
    assert a == c


def test_family_key_never_empty():
    assert splits._family_key("!!!", "US", None)


def test_test_mix_weights_are_derived_not_hard_coded(monkeypatch):
    """Weights come from the test file's country counts, LOCO absorbing the rest."""
    import pandas as pd
    from ber import splits as S

    fake = pd.DataFrame({"country": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
    monkeypatch.setattr("ber.data.load_frame", lambda *a, **k: fake)
    w = S.test_mix_weights(["A", "B"])
    assert w == pytest.approx({"A": 0.5, "B": 0.3, "LOCO": 0.2})
    # a country only in training simply does not appear
    w2 = S.test_mix_weights(["A", "Z"])
    assert set(w2) == {"A", "LOCO"}
    assert w2["LOCO"] == pytest.approx(0.5)


def test_reweighted_falls_back_to_the_plain_mean():
    got = splits.reweighted_score({"Atlantis": 0.8, "Borduria": 0.6}, loco=float("nan"))
    assert got == pytest.approx(0.7)


def test_meta_and_calibration_never_share_a_family(fam):
    """Fold 4 splits between M2 and calibration by family, not by id."""
    import numpy as np
    from ber.run import _split_by_family

    ids = fam["id"].to_numpy(dtype=np.int32)
    a, b = _split_by_family(fam, ids, 0.6)
    assert not (set(a.tolist()) & set(b.tolist()))
    assert len(a) + len(b) == len(ids)
    fam_a = set(fam[fam["id"].isin(a)]["family"])
    fam_b = set(fam[fam["id"].isin(b)]["family"])
    assert not (fam_a & fam_b)
    # and the union is untouched, so cached blocking stays valid
    assert set(np.concatenate([a, b]).tolist()) == set(ids.tolist())


def test_split_by_family_is_deterministic(fam):
    import numpy as np
    from ber.run import _split_by_family
    ids = fam["id"].to_numpy(dtype=np.int32)
    a1, b1 = _split_by_family(fam, ids, 0.5)
    a2, b2 = _split_by_family(fam, ids, 0.5)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
