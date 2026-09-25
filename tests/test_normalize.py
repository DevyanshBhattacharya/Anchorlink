"""Normalisation must be script-agnostic and must not invent language rules."""
import math

import pytest

from ber.normalize import (fold, tokens, clean_text, name_aliases, house_numbers,
                           is_abbrev, skeleton, has_native_script, char_ngrams,
                           TokenRoles, index_view, skeleton_view, is_domain_name)


def test_fold_romanises_and_lowercases():
    assert fold("Président") == "president"
    assert fold("Léarning Center") == "learning center"
    assert "dev" not in fold("राम")          # romanised, not dropped
    assert fold("राम") .strip() != ""


def test_tokens_drop_immediate_repeats():
    assert tokens("Leela Leela Diaz") == ["leela", "diaz"]
    assert tokens("A A A B") == ["a", "b"]
    assert tokens("Alpha Beta Alpha") == ["alpha", "beta", "alpha"]  # non-adjacent kept


def test_placeholders_removed():
    assert clean_text("<< Team Ecole") == "team ecole"
    assert clean_text("-- Northwind Peak Inc") == "northwind peak inc"
    assert clean_text("##8 Foo") == "8 foo"
    assert clean_text("<NULL>") == ""


def test_name_aliases():
    assert name_aliases("8913textiles.com") == ["8913textiles"]
    assert name_aliases("Leela Leela Diaz [Pimco]") == ["leela diaz pimco"]
    a = name_aliases("Acme Corp dba Roadrunner")
    assert "acme corp" in a and "roadrunner" in a
    assert name_aliases("") == []


def test_is_domain_name():
    assert is_domain_name("wilfordhancock.com")
    assert not is_domain_name("Prime Money")


def test_house_numbers():
    assert house_numbers("003182 LYNCHBURG STREET") == {"3182"}
    assert house_numbers("4109-4111 Main") == {"4109", "4111"}
    assert house_numbers("63/2275/7") == {"63", "2275", "7"}
    assert house_numbers("") == set()
    assert house_numbers("No street number here") == set()


def test_abbrev_soft_test():
    assert is_abbrev("rd", "road")
    assert is_abbrev("pvt", "private")
    assert is_abbrev("bd", "boulevard")
    assert is_abbrev("ste", "societe")
    assert is_abbrev("av", "avenue")
    assert not is_abbrev("road", "rd")        # direction matters
    assert not is_abbrev("abc", "xyz")


def test_skeleton_bridges_scripts():
    """The doc's headline example: both spellings collapse to one key."""
    latin = skeleton("Dynamic Hospitality Private Limited")
    deva = skeleton("डायनामिक हॉस्पिटैलिटी प्राइवेट लिमिटेड")
    assert latin == "dnmk hsptlt prvt lmtd"
    assert latin == deva


def test_skeleton_is_stable_under_common_noise():
    assert skeleton("Apex Digital") == skeleton("apex  digital")
    assert skeleton("Pvt Ltd") == skeleton("PVT LTD")


def test_has_native_script():
    assert has_native_script("मॉडर्न फाइनेंस")
    assert not has_native_script("Modern Finance")
    assert not has_native_script("Président")     # Latin-1 accents are not native script


def test_char_ngrams():
    assert char_ngrams("abcd") == {"abc", "bcd"}
    assert char_ngrams("ab") == {"ab"}
    assert char_ngrams("") == set()


def _france_fixture():
    """Legal forms always at the tail; brand tokens spread over all positions."""
    brands = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    names = []
    for i, b in enumerate(brands):
        for j, o in enumerate(brands):
            if b == o:
                continue
            suffix = "SARL" if (i + j) % 2 == 0 else "SAS"
            names.append(f"{b} {o} widgets {suffix}")
            names.append(f"widgets {o} {b} {suffix}")
    return names


def test_token_roles_finds_legal_forms_without_a_wordlist():
    names = _france_fixture()
    roles = TokenRoles(min_share=0.01, edge_ratio=0.85).fit(names, ["France"] * len(names))
    assert "sarl" in roles.affix["France"]
    assert "sas" in roles.affix["France"]
    # brand tokens appear in the middle too, so the edge statistic leaves them alone
    assert "alpha" not in roles.affix["France"]
    assert "widgets" not in roles.affix["France"]
    # damping: an affix weighs a tenth of its IDF
    assert roles.weight("sarl", "France") < roles.weight("alpha", "France")


def test_token_roles_core_name_strips_affixes():
    brands = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    names = []
    for b in brands:
        for o in brands:
            if b == o:
                continue
            names.append(f"{b} {o} tools private limited")
            names.append(f"tools {o} {b} private limited")
    roles = TokenRoles(min_share=0.01, bigram_min_share=0.01).fit(names, ["India"] * len(names))
    assert "limited" in roles.affix["India"]
    core = roles.core_name("alpha beta tools private limited", "India")
    assert "alpha" in core and "beta" in core
    assert "limited" not in core
    assert "private" not in core


def test_token_roles_never_returns_empty_core():
    roles = TokenRoles(min_share=0.01).fit(["Limited"] * 100, ["India"] * 100)
    assert roles.core_tokens("Limited", "India") == ["limited"]


def test_core_backs_off_instead_of_collapsing():
    """A name made only of affixes keeps a core rather than returning nothing."""
    brands = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    names = [f"{b} {o} tools private limited" for b in brands for o in brands if b != o]
    roles = TokenRoles(min_share=0.01, bigram_min_share=0.01).fit(names, ["India"] * len(names))
    assert roles.core_tokens("private limited", "India")     # never empty


def test_token_roles_unseen_country_falls_back():
    roles = TokenRoles().fit(["Alpha Beta Gamma"], ["US"])
    w = roles.weight("anything", "Mars")
    assert w > 0 and w == 10.0                 # the documented fallback


def test_token_roles_roundtrip():
    roles = TokenRoles(min_share=0.05).fit(["Alpha Widgets SARL"] * 40 + ["Beta Tools SAS"] * 40,
                                           ["France"] * 80,
                                           ["1 rue de la paix"] * 80)
    back = TokenRoles.from_dict(roles.to_dict())
    assert back.affix == roles.affix
    assert back.weight("sarl", "France") == roles.weight("sarl", "France")
    assert back.addr_weight("rue", "France") == roles.addr_weight("rue", "France")


def test_views():
    assert index_view("<< Team Ecole", "175 Blvd, Bordeaux") == "team ecole | 175 blvd bordeaux"
    assert skeleton_view("Apex", "1 Road") == f"{skeleton('Apex')} | {skeleton('1 Road')}"


def test_name_freq_pct_ranks_generic_names_higher():
    names = ["Common Name Ltd"] * 500 + [f"Rare Brand {i} Ltd" for i in range(500)]
    roles = TokenRoles(name_hash_slots=1 << 16).fit(names, ["US"] * len(names))
    generic = roles.name_freq_pct("Common Name Ltd", "US")
    rare = roles.name_freq_pct("Rare Brand 7 Ltd", "US")
    assert generic > rare >= 0.0


def test_partial_fit_equals_one_shot():
    names = [f"Brand {i % 37} Widgets SARL" for i in range(400)]
    ctry = ["France"] * 400
    addrs = [f"{i} rue de la paix" for i in range(400)]
    a = TokenRoles(min_share=0.01).fit(names, ctry, addrs, min_df=2)
    b = TokenRoles(min_share=0.01)
    for lo in range(0, 400, 73):
        b.partial_fit(names[lo:lo + 73], ctry[lo:lo + 73], addrs[lo:lo + 73])
    b.finalize(min_df=2)
    assert a.affix == b.affix
    assert a.idf == b.idf
    assert a.n_docs == b.n_docs


def test_finalize_prunes_hapax_losslessly():
    names = ["alpha beta gamma"] * 10 + ["unique%d token" % i for i in range(50)]
    roles = TokenRoles().fit(names, ["US"] * len(names), min_df=2)
    assert "unique0" not in roles.idf["US"]              # pruned
    # ...and the fallback reproduces exactly what log(N/1) would have been
    assert math.isclose(roles.weight("unique0", "US"), math.log(len(names)), rel_tol=1e-12)


def test_subset_keeps_one_country_and_still_works():
    names = ["alpha widgets sarl"] * 40 + ["beta tools llc"] * 40
    ctry = ["France"] * 40 + ["US"] * 40
    addrs = ["1 rue de la paix"] * 40 + ["2 main st"] * 40
    roles = TokenRoles(min_share=0.05).fit(names, ctry, addrs, min_df=1)
    fr = roles.subset("France")
    assert set(fr.idf) == {"France"}
    assert set(fr.n_docs) == {"France"}
    assert fr.weight("widgets", "France") == roles.weight("widgets", "France")
    assert fr.addr_weight("rue", "France") == roles.addr_weight("rue", "France")
    assert fr.core_tokens("alpha widgets sarl", "France") == \
        roles.core_tokens("alpha widgets sarl", "France")
    # an unknown country still falls back cleanly rather than raising
    assert fr.weight("anything", "US") == 10.0


def test_subset_of_a_missing_country_is_empty_but_usable():
    roles = TokenRoles().fit(["a b c"], ["US"])
    other = roles.subset("Mars")
    assert other.idf == {}
    assert other.weight("x", "Mars") == 10.0


@pytest.mark.parametrize("text,expected_label", [
    ("8913textiles.com", "8913textiles"),
    ("wilfordhancock.com", "wilfordhancock"),
    ("sunmaahospitality.com", "sunmaahospitality"),
    ("mabrasserie.fr", "mabrasserie"),
    ("shop.example.co.uk", "example"),
])
def test_domain_names_reduce_to_their_label(text, expected_label):
    assert is_domain_name(text)
    assert expected_label in name_aliases(text)


@pytest.mark.parametrize("text", ["Apex Pvt.Ltd", "Apex Digital", "St. Louis Foods",
                                  "Smt. Radha Traders"])
def test_ordinary_abbreviations_are_not_domains(text):
    """A bare [a-z]{2,6} TLD pattern would read 'pvt.ltd' as a domain."""
    assert not is_domain_name(text)
