"""Unit tests for the TF-IDF similarity module."""

import pytest

from similarity import (
    cosine_similarity,
    keyword_frequency_analysis,
    resume_job_similarity,
    tokenize,
)


@pytest.mark.unit
def test_tokenize_basic():
    tokens = tokenize("Python developer with 5 years experience")
    assert "python" in tokens
    assert "developer" in tokens
    assert "experience" in tokens
    # Stop words removed
    assert "with" not in tokens


@pytest.mark.unit
def test_tokenize_empty():
    assert tokenize("") == []


@pytest.mark.unit
def test_cosine_similarity_identical():
    vec = {"python": 1.0, "java": 0.5}
    assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=0.001)


@pytest.mark.unit
def test_cosine_similarity_orthogonal():
    vec_a = {"python": 1.0}
    vec_b = {"java": 1.0}
    assert cosine_similarity(vec_a, vec_b) == 0.0


@pytest.mark.unit
def test_cosine_similarity_empty():
    assert cosine_similarity({}, {"a": 1.0}) == 0.0
    assert cosine_similarity({}, {}) == 0.0


@pytest.mark.unit
def test_resume_job_similarity():
    resume = "Python developer experienced with Django REST APIs PostgreSQL"
    job = "Looking for Python developer with Django and PostgreSQL experience"
    score = resume_job_similarity(resume, job)
    assert 0.0 < score <= 1.0


@pytest.mark.unit
def test_keyword_frequency():
    results = keyword_frequency_analysis(
        ["Python", "Django", "Java"],
        "Python and Django developer",
        "Python developer with Java and Django experience",
    )
    assert len(results) == 3
    python_result = next(r for r in results if r["keyword"] == "Python")
    assert python_result["resume_count"] >= 1
    assert python_result["job_count"] >= 1
