"""Unit tests for the shared match-score formula (scoring.py)."""

import pytest

from scoring import compute_match_score


@pytest.mark.unit
class TestComputeMatchScore:
    """Bounds, weighting, and known values."""

    def test_known_value_gap_analysis_style_inputs(self):
        # cos 0.35 -> 14.0, coverage 0.9 -> 36.0, +20 base = 70.0
        assert compute_match_score(0.35, 0.9) == 70.0

    def test_perfect_inputs(self):
        assert compute_match_score(1.0, 1.0) == 100.0

    def test_zero_inputs_still_get_base(self):
        assert compute_match_score(0.0, 0.0) == 20.0

    def test_clamped_to_100_on_overshoot(self):
        assert compute_match_score(1.5, 1.2) == 100.0

    def test_clamped_to_0_on_negative_inputs(self):
        assert compute_match_score(-0.5, -1.0) == 0.0

    def test_rounding_to_one_decimal(self):
        # 13.33... + 26.66... + 20 = 60.0 after rounding
        assert compute_match_score(1 / 3, 2 / 3) == 60.0

    def test_symmetric_weighting(self):
        # cos and coverage contribute equally: swapping them changes nothing
        assert compute_match_score(0.8, 0.4) == compute_match_score(0.4, 0.8)
