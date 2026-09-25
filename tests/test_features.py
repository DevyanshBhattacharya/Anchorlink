"""Features must be country-agnostic, NaN-safe and behave the way the doc claims."""
import math

import numpy as np
import pytest

from ber.features import (META_FEATURES, N_PAIR, PAIR_FEATURES, RecordView,
                          build_meta_features, jaccard, monge_elkan,
                          monotone_vector, number_relation, pair_feature_row,
                          soft_tfidf, group_starts, rank_within_groups,
                          best_other_source)
from ber.normalize import TokenRoles


@pytest.fixture(scope="module")
def real_roles():
    """The corpus statistics actually fitted on the training files, when present."""
    import pickle
    from ber.corpus import roles_path
    path = roles_path("train")
    if not path.exists():
        pytest.skip("train corpus statistics not built yet")
    with open(path, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def roles():
    brands = ["apex", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    names, ctry, addr = [], [], []
    for b in brands:
        for o in brands:
            if b == o:
                continue
            names += [f"{b} {o} digital llc", f"digital {o} {b} llc"]
            ctry += ["US", "US"]
            addr += ["1795 westchester drive high point nc", "17560 ellis road tahlequah ok"]
    # an Indian partition too, so the legal-form tests have private/limited/ltd.
    # Addresses must vary or every token's IDF collapses to zero.
    cities = ["ernakulam", "jaipur", "kochi", "bhopal", "gurgaon", "nagpur", "indore"]
    for i, b in enumerate(brands):
        for j, o in enumerate(brands):
            if b == o:
                continue
            for k, suffix in enumerate(("private limited", "limited", "ltd")):
                names.append(f"{b} {o} traders {suffix}")
                ctry.append("India")
                addr.append(f"{10 + i * 7 + j} mg road {cities[(i + j + k) % len(cities)]}")
    return TokenRoles(min_share=0.01, bigram_min_share=0.01).fit(names, ctry, addr, min_df=1)


def feats(roles, n1, a1, n2, a2, country="US"):
    v1 = RecordView(n1, a1, country, roles)
    v2 = RecordView(n2, a2, country, roles)
    out = np.empty(N_PAIR, dtype=np.float64)
    pair_feature_row(v1, v2, roles, country, out)
    return dict(zip(PAIR_FEATURES, out))


def test_identical_records_score_at_the_top(roles):
    f = feats(roles, "Apex Digital LLC", "1795 Brackendale Drive, High Point, NC",
              "Apex Digital LLC", "1795 Brackendale Drive, High Point, NC")
    assert f["name_jw"] == 1.0
    assert f["name_lev"] == 1.0
    assert f["addr_contain"] == pytest.approx(1.0)
    assert f["num_rel"] == 1.0
    assert f["addr_unmatched_1"] == pytest.approx(0.0)


def test_zero_padded_number_is_an_exact_match(roles):
    f = feats(roles, "Apex Digital LLC", "3182 Ashgrove Street",
              "APEX DIGITAL", "003182 LYNCHBURG STREET")
    assert f["num_rel"] == 1.0          # class 1: a number in common


def test_neighbouring_house_number_is_the_hard_negative(roles):
    """The dominant same-name hard negative: same street, a few doors away."""
    f = feats(roles, "Apex Digital LLC", "616 Kingsmere Avenue, Columbus, OH",
              "Apex Digital LLC", "627 Kingsmere Avenue, Columbus, OH")
    assert f["num_rel"] == 4.0          # class 4: differ by 3-20
    assert f["name_jw"] == 1.0          # names give no evidence at all
    assert f["num_min_diff"] == pytest.approx(math.log1p(11))


def test_number_truncation_class(roles):
    f = feats(roles, "X Ltd", "1447 Harborview Blvd", "X Ltd", "447 Harborview Blvd")
    assert f["num_rel"] == 2.0


def test_missing_address_is_nan_not_zero(roles):
    f = feats(roles, "Apex Digital LLC", "1795 Brackendale Drive", "Apex Digital LLC", "")
    assert f["addr_missing"] == 1.0
    assert math.isnan(f["addr_contain"])
    assert math.isnan(f["num_min_diff"])
    assert f["num_rel"] == 0.0          # class 0: missing


def test_native_script_name_is_bridged_by_the_skeleton(roles):
    f = feats(roles, "Dynamic Hospitality Private Limited", "12 MG Road",
              "डायनामिक हॉस्पिटैलिटी प्राइवेट लिमिटेड", "12 MG Road", country="India")
    assert f["name_skel_jac"] == pytest.approx(1.0)
    assert f["name_gram_jac"] < 0.8      # the romanised form alone matches poorly
    assert f["native_flag"] == 1.0


def test_abbreviation_is_partial_not_equal(roles):
    f = feats(roles, "Apex Digital Pvt Ltd", "1 Rd", "Apex Digital Private Limited", "1 Road")
    assert 0.0 < f["addr_me"] <= 1.0
    assert f["name_me_12"] > 0.8


def test_containment_beats_jaccard_on_one_sided_landmarks(roles):
    short = feats(roles, "A Ltd", "12 MG Road", "A Ltd", "12 MG Road", country="India")
    longer = feats(roles, "A Ltd", "12 MG Road", "A Ltd",
                   "12 MG Road, Near Fortis Hospital, Opp SBI ATM, Bengaluru", country="India")
    assert longer["addr_contain"] == pytest.approx(short["addr_contain"], abs=1e-6)
    assert longer["addr_unmatched_2"] > longer["addr_unmatched_1"]


def test_domain_alias(roles):
    f = feats(roles, "8913 Textiles", "1 Main St", "8913textiles.com", "1 Main St")
    assert f["domain_flag"] == 1.0
    assert f["alias_best"] > 0.8


def test_the_dropped_word_that_every_other_feature_missed(real_roles):
    """The worked example from the training data, looked up by id.

    S1-499359562 has three true matches in the ground truth, all of which keep a
    word that a fourth record — which matches no S1 entity at all — drops.  Three
    features are blind to it: the core name strips it (the affix bigram is
    flagged), `token_set_ratio` returns 100 for a subset, and its IDF is too low
    to move the weighted mass.  The one-sided token count has to carry it.

    The records are read from the dataset rather than written down here, so no
    challenge data lives in this repository.
    """
    from ber import config
    from ber.io_utils import read_ground_truth

    if not config.FILES[("train", "S1")].is_file():
        pytest.skip("training data not present")

    s1_id, distractor_id = "S1-499359562", "S2-83808461"
    gt = read_ground_truth()
    truth_ids = gt.get(s1_id)
    if not truth_ids:
        pytest.skip("this example is not in the provided ground truth")
    assert distractor_id not in truth_ids, "the distractor must not be a true match"

    rec = _lookup_records({s1_id, distractor_id, *truth_ids})
    if len(rec) != 1 + 1 + len(truth_ids):
        pytest.skip("could not read every record of the example")

    s1_name, s1_addr = rec[s1_id]

    def f(cand_id):
        n, a = rec[cand_id]
        return feats(real_roles, s1_name, s1_addr, n, a, country="India")

    for cand in truth_ids:
        assert f(cand)["name_tok_only_1"] == 0.0, cand
    assert f(distractor_id)["name_tok_only_1"] >= 1.0

    # and confirm the features that miss it really do miss it
    for cand in list(truth_ids) + [distractor_id]:
        g = f(cand)
        assert g["name_tset"] == pytest.approx(1.0), cand
        assert g["name_idf_unmatched_1"] == pytest.approx(0.0, abs=1e-6), cand


def _lookup_records(ids):
    """{entity_id: (name, address)} read straight from the source TSVs."""
    from ber import config
    from ber.io_utils import READ_KW
    import pandas as pd

    want = set(ids)
    out = {}
    for src in ("S1", "S2", "S3"):
        need = {i for i in want if i.startswith(src + "-")}
        if not need:
            continue
        path = config.FILES[("train", src)]
        for chunk in pd.read_csv(path, chunksize=500_000, **READ_KW):
            hit = chunk[chunk["entity_id"].isin(need)]
            for _, row in hit.iterrows():
                out[row["entity_id"]] = (row["business_name"], row["business_address"])
            need -= set(out)
            if not need:
                break
    return out


def test_ltd_and_limited_are_not_a_missing_word(real_roles):
    a = feats(real_roles, "Alpha Traders Limited", "1 MG Road",
              "Alpha Traders Ltd", "1 MG Road", country="India")
    assert a["name_tok_only_1"] == 0.0 and a["name_tok_only_2"] == 0.0


def test_token_features_on_identical_names(roles):
    addr = "12 MG Road, Ernakulam"
    identical = feats(roles, "Alpha Traders Private Limited", addr,
                      "Alpha Traders Private Limited", addr, country="India")
    assert identical["name_tok_jac"] == pytest.approx(1.0)
    assert identical["name_tok_only_1"] == 0.0 and identical["name_tok_only_2"] == 0.0
    assert identical["name_tok_eq"] == 1.0


def test_extra_word_is_counted_on_the_right_side(roles):
    addr = "12 MG Road, Ernakulam"
    f = feats(roles, "Alpha Beta Traders Private Limited", addr,
              "Alpha Traders Private Limited", addr, country="India")
    assert f["name_tok_only_1"] >= 1.0
    assert f["name_tok_only_2"] == 0.0
    g = feats(roles, "Alpha Traders Private Limited", addr,
              "Alpha Beta Traders Private Limited", addr, country="India")
    assert g["name_tok_only_2"] >= 1.0
    assert g["name_tok_only_1"] == 0.0


def test_word_order_change_is_not_a_missing_word(roles):
    """'LLC Anchor Angel' vs 'Anchor Angel LLC' is a move, not a dropped token."""
    f = feats(roles, "Anchor Angel LLC", "1 Main St", "LLC Anchor Angel", "1 Main St")
    assert f["name_tok_jac"] == pytest.approx(1.0)
    assert f["name_tok_only_1"] == 0.0 and f["name_tok_only_2"] == 0.0
    assert f["name_tok_eq"] == 1.0


def test_no_feature_encodes_the_country(roles):
    """The same pair in two countries differs only through corpus statistics."""
    assert "country" not in " ".join(PAIR_FEATURES)
    assert len(monotone_vector(PAIR_FEATURES)) == N_PAIR


def test_number_relation_classes():
    assert number_relation(set(), {"1"}) == 0
    assert number_relation({"1795"}, {"1795"}) == 1
    assert number_relation({"1447"}, {"447"}) == 2
    assert number_relation({"616"}, {"617"}) == 3
    assert number_relation({"616"}, {"627"}) == 4
    assert number_relation({"616"}, {"9999"}) == 5


def test_jaccard_and_monge_elkan_nan_on_empty():
    assert math.isnan(jaccard(set(), {"a"}))
    assert math.isnan(monge_elkan([], ["a"], {}))
    assert math.isnan(soft_tfidf([], {}, ["a"], {}))


def test_meta_features_shapes_and_exclusivity():
    s1 = np.array([0, 0, 0, 1, 1])
    cand = np.array([10, 11, 12, 10, 13])      # candidate 10 is claimed twice
    src = np.array([2, 3, 2, 2, 3], dtype=np.int8)
    p = np.array([0.9, 0.4, 0.2, 0.6, 0.8])
    M = build_meta_features(s1, cand, src, p)
    assert M.shape == (5, len(META_FEATURES))
    cols = dict(zip(META_FEATURES, M.T))
    assert list(cols["m_rank"]) == [0, 1, 2, 1, 0]
    assert cols["m_n_competitors"][0] == 2 and cols["m_n_competitors"][1] == 1
    # the doc's worked example: 0.90 vs 0.60 on the same record -> 0.78 / 0.13
    assert cols["m_excl_prob"][0] == pytest.approx(0.78, abs=0.01)
    assert cols["m_excl_prob"][3] == pytest.approx(0.13, abs=0.01)


def test_group_helpers():
    g = np.array([0, 0, 1, 1, 1, 2])
    b = group_starts(g)
    assert list(b) == [0, 2, 5, 6]
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.2, 1.0])
    assert list(rank_within_groups(b, s)) == [1, 0, 1, 0, 2, 0]
    src = np.array([2, 3, 2, 3, 2, 2], dtype=np.int8)
    bo = best_other_source(b, s, src)
    assert bo[0] == pytest.approx(0.9)       # row 0 is S2, best S3 in group is 0.9
    assert bo[1] == pytest.approx(0.1)


# --------------------------------------------------------------- v3 features
def test_name_written_into_the_address_field_is_visible(roles):
    """The cross-field columns are the only ones that can see this pair.

    A recurring shape in the data: the candidate's name is a fragment while the
    business name sits in its address line.  Name-to-name and address-to-address
    comparisons both read a conflict; the cross-field mass is what carries the
    evidence.
    """
    f = feats(roles, "apex beta digital llc", "1795 westchester drive high point nc",
              "unknown", "apex beta digital 1795 westchester drive high point nc")
    assert f["xf_n1_a2"] > 0.5, "S1's name mass should be found in the candidate address"
    assert f["name_jw"] < 0.7, "the names themselves do not match"


def test_cross_field_is_nan_when_the_other_side_has_no_address(roles):
    f = feats(roles, "apex beta digital llc", "1795 westchester drive high point nc",
              "apex beta digital llc", "")
    assert math.isnan(f["xf_n1_a2"])


def test_rare_shared_token_outranks_a_common_one(roles):
    """A match on a rare word is worth more than a match on a frequent one.

    Both pairs below share exactly one non-affix token and differ in the other, so
    every unweighted overlap feature is identical; only the rarity column moves.
    """
    common = feats(roles, "apex digital llc", "x", "beta digital llc", "x")
    rare = feats(roles, "apex digital llc", "x", "apex gamma llc", "x")
    assert rare["name_rare_shared"] > common["name_rare_shared"]
    assert 0.0 <= common["name_rare_shared"] <= 1.0
    assert 0.0 <= rare["name_rare_shared"] <= 1.0


def test_skeleton_edit_distance_separates_a_transposition(roles):
    """Gram Jaccard is order-free, so it cannot; the edit distance can."""
    same = feats(roles, "telecommunication", "x", "tetlecommunication", "x")
    shuffled = feats(roles, "telecommunication", "x", "communicationtele", "x")
    assert same["name_skel_lev"] > shuffled["name_skel_lev"]


def test_administrative_tail_forgives_a_mangled_street(roles):
    """Same city and state, unrecognisable street: the tail is the only agreement."""
    f = feats(roles, "apex beta digital llc", "1795 westchester drive high point nc",
              "apex beta digital llc", "1795 wstchstr dr high point nc")
    assert f["adm_tail_jac"] > 0.5
    assert f["adm_tail_cov"] > 0.5


def test_administrative_tail_survives_a_reordered_address(roles):
    """``adm_tail_cov`` compares against the whole other address, so a rotated
    field ("NC, High Point, 1795 Westchester Drive") still agrees where the
    strict tail-to-tail Jaccard does not."""
    f = feats(roles, "apex beta digital llc", "1795 westchester drive high point nc",
              "apex beta digital llc", "nc high point 1795 westchester drive")
    assert f["adm_tail_cov"] == pytest.approx(1.0)


def test_pin_match_is_three_valued(roles):
    hit = feats(roles, "apex llc", "12 mg road kochi 682024",
                "apex llc", "12 mg road kochi 682024", country="India")
    miss = feats(roles, "apex llc", "12 mg road kochi 682024",
                 "apex llc", "12 mg road kochi 560001", country="India")
    absent = feats(roles, "apex llc", "12 mg road kochi",
                   "apex llc", "12 mg road kochi", country="India")
    assert hit["adm_pin_match"] == 1.0
    assert miss["adm_pin_match"] == 0.0
    assert math.isnan(absent["adm_pin_match"]), "no PIN is not a conflict"


def test_address_skeleton_bridges_a_native_script_address(roles):
    """Devanagari and its romanisation share no token, so every other address
    feature reads a conflict.  The address skeleton is the bridge, exactly as the
    name skeleton is for names."""
    f = feats(roles, "apex traders limited", "कोरमंगला",
              "apex traders ltd", "koramangala", country="India")
    assert f["addr_skel_jac"] > 0.0


def test_v3_meta_columns_describe_the_list(roles):
    s1 = np.array([0, 0, 0, 1, 1])
    cand = np.array([10, 11, 12, 10, 13])
    src = np.array([2, 3, 2, 2, 3], dtype=np.int8)
    p = np.array([0.9, 0.4, 0.2, 0.6, 0.8])
    cols = dict(zip(META_FEATURES, build_meta_features(s1, cand, src, p).T))
    # entity 0: best 0.9, runner-up 0.4 -> a steep cliff, shared by every row
    assert cols["m_cliff"][0] == pytest.approx((0.9 - 0.4) / 0.9)
    assert cols["m_cliff"][1] == cols["m_cliff"][0]
    assert cols["m_gap_rel"][0] == pytest.approx(0.0)
    assert cols["m_psum"][0] == pytest.approx(1.5)
    assert cols["m_std"][0] == pytest.approx(np.std([0.9, 0.4, 0.2]))
    # rank inside the row's own source: entity 0 has S2 = {0.9, 0.2}, S3 = {0.4}
    assert list(cols["m_rank_src"][:3]) == [0, 0, 1]
    # candidate 10 is contested, candidate 13 is not
    assert cols["m_excl_loss"][0] > 0.0
    assert cols["m_excl_loss"][4] == pytest.approx(0.0, abs=1e-9)


def test_new_columns_all_have_a_monotone_opinion():
    """A similarity that is allowed to cut both ways will not transfer to a
    country the model never saw, so each new column declares a direction."""
    from ber.features import monotone_vector
    mono = dict(zip(PAIR_FEATURES, monotone_vector(PAIR_FEATURES)))
    for f in ("xf_n1_a2", "xf_n2_a1", "name_idf_jac", "name_rare_shared",
              "name_skel_lev", "adm_tail_jac", "adm_tail_cov", "adm_pin_match",
              "addr_skel_jac", "addr_rare_shared"):
        assert mono[f] == 1, f
