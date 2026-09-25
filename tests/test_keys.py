"""Key blocks must be capped, stable and reachable in both directions."""
import numpy as np

from ber.keys import KeyBlocks, record_keys, hash64
from ber.normalize import TokenRoles


def _roles():
    brands = ["apex", "beta", "gamma", "delta", "epsilon", "zeta"]
    names = [f"{b} {o} digital llc" for b in brands for o in brands if b != o]
    return TokenRoles(min_share=0.01, bigram_min_share=0.01).fit(names, ["US"] * len(names))


def test_hash64_fits_int64_and_is_stable():
    for s in ("a", "n:1795|pks", "s:dgtl pks", "राम"):
        h = hash64(s)
        assert -(1 << 63) <= h < (1 << 63)
        assert h == hash64(s)
    assert np.asarray([hash64("x")], dtype=np.int64)[0] == hash64("x")


def test_zero_padding_and_typo_share_a_key():
    roles = _roles()
    a = record_keys("Apex Digital LLC", "1795 Westchester Dr", "US", roles)
    b = record_keys("Apex Digital LLC", "001795 Westchester Drive", "US", roles)
    assert set(a) & set(b)              # leading zeros stripped -> same number key


def test_lookup_finds_the_right_rows():
    roles = _roles()
    names = ["Apex Digital LLC", "Beta Tools LLC", "Apex Digital LLC"]
    addrs = ["1795 Westchester Dr", "12 Other St", "001795 Westchester Drive"]
    kb = KeyBlocks.build(names, addrs, "US", roles)
    hits = set(kb.lookup(record_keys(names[0], addrs[0], "US", roles)).tolist())
    assert {0, 2} <= hits
    assert 1 not in hits


def test_blocks_are_capped():
    roles = _roles()
    n = 200
    names = ["Apex Digital LLC"] * n
    addrs = ["1795 Westchester Dr"] * n
    kb = KeyBlocks.build(names, addrs, "US", roles, max_block=50)
    counts = np.diff(kb.starts)
    assert counts.max() <= 50
    assert len(kb.lookup(record_keys(names[0], addrs[0], "US", roles))) <= 100  # 2 keys x 50


def test_empty_input():
    roles = _roles()
    kb = KeyBlocks.build([], [], "US", roles)
    assert len(kb.lookup([hash64("anything")])) == 0


def test_address_without_numbers_still_has_a_skeleton_key():
    roles = _roles()
    keys = record_keys("Apex Digital LLC", "", "US", roles)
    assert len(keys) == 1
