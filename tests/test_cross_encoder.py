"""Augmentation operators and the uncertain band."""
import random

import numpy as np
import pytest

from ber.cross_encoder import (CrossEncoderConfig, JUDGE_INSTRUCTION, corrupt,
                               pair_text, shift_house_number, uncertain_band)
from ber.normalize import house_numbers


def test_shift_house_number_moves_it_into_the_hard_negative_band():
    rng = random.Random(0)
    for _ in range(50):
        out = shift_house_number("616 Kingsmere Avenue, Columbus, OH", rng)
        a = house_numbers("616 Kingsmere Avenue")
        b = house_numbers(out)
        d = min(abs(int(x) - int(y)) for x in a for y in b)
        assert 3 <= d <= 20
        assert "Kingsmere Avenue" in out      # only the number changes


def test_shift_house_number_returns_none_without_a_number():
    assert shift_house_number("Kingsmere Avenue, Columbus", random.Random(0)) is None
    assert shift_house_number("", random.Random(0)) is None


def test_shift_never_produces_a_non_positive_number():
    rng = random.Random(1)
    for _ in range(50):
        out = shift_house_number("2 Main St", rng)
        assert all(int(n) >= 1 for n in house_numbers(out))


def test_corrupt_is_label_preserving_in_shape():
    rng = random.Random(3)
    src = "Apex Digital LLC 1795 Brackendale Drive High Point"
    seen = set()
    for _ in range(40):
        out = corrupt(src, rng)
        assert isinstance(out, str) and out
        seen.add(out)
    assert len(seen) > 3                    # actually varies


def test_uncertain_band_is_the_middle():
    p = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0])
    assert list(uncertain_band(p)) == [False, False, True, True, True, False, False]


def test_pair_text_puts_numbers_in_their_own_field():
    t = pair_text("Apex Digital", "616 Kingsmere Avenue", "US")
    nums = t.split("nums:")[1].split("|")[0]
    assert "616" in nums
    assert t.index("name:") < t.index("nums:") < t.index("addr:")


def test_judge_instruction_states_the_number_rule():
    assert "house number" in JUDGE_INSTRUCTION
    assert "different business" in JUDGE_INSTRUCTION


def test_config_shrinks_off_cuda():
    big = CrossEncoderConfig.for_device("cuda")
    small = CrossEncoderConfig.for_device("cpu")
    assert small.batch_size < big.batch_size
    assert small.max_length <= big.max_length
