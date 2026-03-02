"""Integration tests for core API endpoints."""

import io
from unittest.mock import patch

import pytest


@pytest.mark.integration
def test_tailor_no_resume(client):
    with patch("app.services.admin_config.AdminConfigManager.load") as m:
        m.return_value.api_key = "sk-test"
        m.return_value.allow_user_model_selection = False
        m.return_value.default_model = "gpt-4o-mini"
        m.return_value.rate_limit_per_hour = 100
        resp = client.post("/api/tailor", data={"job_description": "x" * 100})
    assert resp.status_code == 400
    assert "resume" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_tailor_short_jd(client):
    with patch("app.services.admin_config.AdminConfigManager.load") as m:
        m.return_value.api_key = "sk-test"
        m.return_value.allow_user_model_selection = False
        m.return_value.default_model = "gpt-4o-mini"
        m.return_value.rate_limit_per_hour = 100
        data = {
            "resume": (io.BytesIO(b"test resume"), "resume.txt"),
            "job_description": "too short",
        }
        resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "short" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_tailor_invalid_mode(client):
    with patch("app.services.admin_config.AdminConfigManager.load") as m:
        m.return_value.api_key = "sk-test"
        m.return_value.allow_user_model_selection = False
        m.return_value.default_model = "gpt-4o-mini"
        m.return_value.rate_limit_per_hour = 100
        data = {
            "resume": (io.BytesIO(b"test resume"), "resume.txt"),
            "job_description": "x" * 100,
            "mode": "invalid",
        }
        resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "mode" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_result_not_found(client):
    resp = client.get("/api/result/nonexistent123")
    assert resp.status_code == 404


@pytest.mark.integration
def test_progress_not_found(client):
    resp = client.get("/api/progress/nonexistent123")
    assert resp.status_code == 404
