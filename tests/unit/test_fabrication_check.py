"""Unit tests for the deterministic fabrication guard."""

import pytest

from fabrication_check import check_bullet, check_bullets


@pytest.mark.unit
class TestCheckBullet:
    """Hard-fact survival between original and optimised bullets."""

    def test_clean_pass_all_facts_survive(self):
        original = "Grew revenue 35% to $1.2M in 2021, a 3x improvement"
        optimised = "Drove 3x revenue growth of 35%, reaching $1,200,000 in 2021"
        assert check_bullet(original, optimised) == []

    def test_missing_percentage_flagged(self):
        assert check_bullet("Improved conversion by 35%", "Improved conversion rates") == ["35%"]

    def test_decimal_percentage_survives(self):
        assert check_bullet("Cut latency 12.5% with caching", "Reduced latency by 12.5% via caching") == []

    def test_missing_currency_flagged(self):
        assert check_bullet("Managed a $500k budget", "Managed departmental budgets") == ["$500k"]

    def test_k_suffix_matches_full_number(self):
        assert check_bullet("Managed a $500k budget", "Owned a $500,000 annual budget") == []

    def test_full_number_matches_k_suffix(self):
        assert check_bullet("Saved $10,000 on tooling", "Negotiated $10k in tooling savings") == []

    def test_million_suffix_equivalence(self):
        assert check_bullet("Closed $1.5m in new ARR", "Generated $1,500,000 of new ARR") == []

    def test_thousands_separator_equivalence(self):
        assert check_bullet("Cut infrastructure spend by $10,000", "Trimmed infra costs $10000") == []

    def test_multiplier_survival(self):
        assert check_bullet("Scaled deployment frequency 3x", "Grew deployment frequency 3x year over year") == []
        assert check_bullet("Scaled deployment frequency 3x", "Increased deployment frequency") == ["3x"]

    def test_year_survival(self):
        assert check_bullet("Rebuilt the platform in 2019", "Led the 2019 platform rebuild") == []
        assert check_bullet("Rebuilt the platform in 2019", "Led a major platform rebuild") == ["2019"]

    def test_plus_sign_stripped_from_counts(self):
        assert check_bullet("Led 20+ engineers across 3 time zones", "Directed 20 engineers") == []

    def test_multiple_violations_aggregated(self):
        original = "Grew ARR 40% to $2m in 2021 while improving efficiency 4x"
        optimised = "Improved annual recurring revenue and team efficiency"
        missing = check_bullet(original, optimised)
        assert set(missing) == {"40%", "$2m", "2021", "4x"}

    def test_empty_original_has_no_facts(self):
        assert check_bullet("", "anything at all") == []

    def test_identical_texts_pass(self):
        text = "Boosted sales 25% to $3M in 2020"
        assert check_bullet(text, text) == []


@pytest.mark.unit
class TestCheckBullets:
    """Bulk API and report shape."""

    def test_report_counts_and_dict_shape(self):
        pairs = [
            (0, 0, "Cut costs 40%", "Reduced spend"),
            (1, 2, "Shipped 12 releases", "Delivered 12 product releases"),
        ]
        report = check_bullets(pairs)
        assert report.checked == 2
        assert report.flagged_count == 1

        data = report.to_dict()
        assert data["checked"] == 2
        assert data["flagged_count"] == 1
        assert len(data["violations"]) == 1
        violation = data["violations"][0]
        assert violation["role_index"] == 0
        assert violation["bullet_index"] == 0
        assert violation["missing_facts"] == ["40%"]
        assert violation["original_text"] == "Cut costs 40%"
        assert violation["optimised_text"] == "Reduced spend"

    def test_empty_input_report(self):
        report = check_bullets([])
        assert report.checked == 0
        assert report.flagged_count == 0
        assert report.to_dict() == {"checked": 0, "flagged_count": 0, "violations": []}
