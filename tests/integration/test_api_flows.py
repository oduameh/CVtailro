"""Integration tests for API flows — tailor, result, history, auth."""

import io
from unittest.mock import patch

import pytest

from app.models import TailoringJob, User


@pytest.fixture
def user(db):
    u = User(
        google_id="test-google-123",
        email="user@example.com",
        name="Test User",
        is_admin=False,
    )
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user):
    with client.session_transaction() as sess:
        sess["_user_id"] = user.id
        sess["_fresh"] = True


@pytest.mark.integration
class TestTailorAPI:
    """Tailor endpoint — validation and flow."""

    def test_tailor_no_resume_returns_400(self, client):
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            resp = client.post("/api/tailor", data={"job_description": "x" * 100})
        assert resp.status_code == 400
        assert "resume" in resp.get_json()["error"].lower()

    def test_tailor_short_jd_returns_400(self, client):
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

    def test_tailor_invalid_mode_returns_400(self, client):
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

    def test_tailor_valid_request_returns_job_id(self, client):
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            with patch("app.services.pipeline.run_pipeline_job"):
                with patch("threading.Thread") as mock_thread:
                    mock_thread.return_value.start = lambda: None
                    data = {
                        "resume": (io.BytesIO(b"x" * 500), "resume.txt"),
                        "job_description": "x" * 100,
                    }
                    resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data
        assert len(data["job_id"]) == 16


@pytest.mark.integration
class TestResultAndProgressAPI:
    """Result and progress endpoints."""

    def test_result_not_found(self, client):
        resp = client.get("/api/result/nonexistent123")
        assert resp.status_code == 404

    def test_progress_not_found(self, client):
        resp = client.get("/api/progress/nonexistent123")
        assert resp.status_code == 404

    def test_result_from_db_job(self, client, user, db):
        job = TailoringJob(
            id="testjob1234567890ab",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume",
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)
        resp = client.get(f"/api/result/{job.id}")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "complete"


@pytest.mark.integration
class TestHistoryAPI:
    """History and saved resumes flow."""

    def test_history_requires_auth(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 401

    def test_history_returns_user_jobs(self, client, user, db):
        job = TailoringJob(
            id="histjob1234567890ab",
            user_id=user.id,
            status="complete",
            job_title="Engineer",
            company="Acme",
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)
        resp = client.get("/api/history")
        assert resp.status_code == 200
        jobs = resp.get_json()["jobs"]
        assert len(jobs) >= 1
        assert jobs[0]["job_title"] == "Engineer"

    def test_saved_resumes_crud_flow(self, client, user):
        _login(client, user)
        resp = client.post(
            "/api/saved-resumes",
            json={"resume_text": "My resume content here", "name": "Resume 1"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        rid = resp.get_json()["id"]
        resp = client.get(f"/api/saved-resumes/{rid}")
        assert resp.status_code == 200
        assert resp.get_json()["resume_text"] == "My resume content here"
        resp = client.delete(f"/api/saved-resumes/{rid}")
        assert resp.status_code == 200
        resp = client.get(f"/api/saved-resumes/{rid}")
        assert resp.status_code == 404


@pytest.mark.integration
class TestHealthAndStatus:
    """Health and status endpoints."""

    def test_health_returns_status(self, client):
        resp = client.get("/api/health")
        assert resp.status_code in (200, 503)
        assert "status" in resp.get_json()
        assert "database" in resp.get_json()

    def test_status_returns_no_secrets(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sk-" not in str(data)
        assert "sk_" not in str(data)
