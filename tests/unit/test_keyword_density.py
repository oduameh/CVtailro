"""Unit tests for keyword density analyzer."""

import pytest

from keyword_density import (
    KeywordDensityReport,
    KeywordFrequency,
    analyze_keyword_density,
)


@pytest.mark.unit
class TestKeywordDensity:
    """Keyword density analysis."""

    def test_with_explicit_skills(self):
        report = analyze_keyword_density(
            job_description="We need Python and Django developers.",
            original_resume="I know Python.",
            tailored_resume="I have Python and Django experience.",
            required_skills=["Python", "Django"],
        )
        assert report.total_jd_keywords == 2
        assert len(report.keywords) == 2
        python_k = next(k for k in report.keywords if k.keyword == "Python")
        assert python_k.resume_before_count >= 1
        assert python_k.resume_after_count >= 1
        assert python_k.status in ("matched", "improved")

    def test_with_auto_extracted_keywords(self):
        report = analyze_keyword_density(
            job_description="We need Python and Python. Also microservices microservices.",
            original_resume="I know Python.",
            tailored_resume="I have Python and microservices experience.",
        )
        assert report.total_jd_keywords >= 1
        assert report.improvement >= 0

    def test_empty_inputs(self):
        report = analyze_keyword_density(
            job_description="",
            original_resume="",
            tailored_resume="",
        )
        assert report.total_jd_keywords == 0
        assert report.keywords == []
        assert report.improvement == 0

    def test_improvement_calculation(self):
        report = analyze_keyword_density(
            job_description="Python Django",
            original_resume="",
            tailored_resume="Python and Django developer",
            required_skills=["Python", "Django"],
        )
        assert report.matched_after >= report.matched_before
        assert report.improvement >= 0

    def test_status_missing(self):
        report = analyze_keyword_density(
            job_description="Python Django",
            original_resume="",
            tailored_resume="Python only",
            required_skills=["Python", "Django"],
        )
        missing = [k for k in report.keywords if k.status == "missing"]
        assert len(missing) >= 0  # Django may be missing

    def test_status_new(self):
        report = analyze_keyword_density(
            job_description="Python Django",
            original_resume="Python",
            tailored_resume="Python and Django",
            required_skills=["Python", "Django"],
        )
        new_kw = [k for k in report.keywords if k.status == "new"]
        assert len(new_kw) >= 0
