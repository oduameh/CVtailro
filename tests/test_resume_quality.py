"""Unit tests for the resume quality analyzer."""

import pytest

from resume_quality import analyze_bullet, analyze_resume, extract_bullets_from_markdown


@pytest.mark.unit
def test_analyze_bullet_with_metrics():
    result = analyze_bullet("Increased revenue by 45% through pipeline automation")
    assert result.has_metrics is True
    assert result.score >= 5


@pytest.mark.unit
def test_analyze_bullet_weak_verb():
    result = analyze_bullet("Helped with the project management tasks")
    assert result.verb_strength == "weak"
    assert result.score < 5


@pytest.mark.unit
def test_analyze_bullet_strong_verb():
    result = analyze_bullet("Spearheaded migration of legacy system to microservices")
    assert result.verb_strength == "strong"


@pytest.mark.unit
def test_analyze_resume_empty():
    report = analyze_resume([])
    assert report.total_bullets == 0
    assert report.overall_score == 0


@pytest.mark.unit
def test_analyze_resume_basic():
    bullets = [
        "Led team of 5 engineers to deliver project 20% ahead of schedule",
        "Helped with documentation",
        "Implemented CI/CD pipeline reducing deployment time by 60%",
    ]
    report = analyze_resume(bullets)
    assert report.total_bullets == 3
    assert report.bullets_with_metrics >= 2
    assert report.overall_score > 0


@pytest.mark.unit
def test_extract_bullets_from_markdown():
    md = """# Resume
## Experience
- Led team of engineers
- Built REST API
* Managed project
"""
    bullets = extract_bullets_from_markdown(md)
    assert len(bullets) == 3
