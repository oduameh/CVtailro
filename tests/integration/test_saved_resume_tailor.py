"""Integration tests for tailoring from a saved resume + rename-without-text."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.models import SavedResume, User


@pytest.fixture
def user(db):
    u = User(google_id="sr-g-1", email="sr@example.com", name="SR User", is_admin=False)
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def other_user(db):
    u = User(google_id="sr-g-2", email="sr2@example.com", name="Other", is_admin=False)
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user):
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user)


def _make_saved(db, user, text="# Jane Doe\nSenior Python Flask engineer", name="Master"):
    r = SavedResume(user_id=user.id, name=name, resume_text=text)
    db.session.add(r)
    db.session.commit()
    return r


def _config_mock(m):
    m.return_value.api_key = "sk-test"
    m.return_value.allow_user_model_selection = False
    m.return_value.default_model = "gpt-4o-mini"
    m.return_value.rate_limit_per_hour = 100
    m.return_value.daily_job_limit = 0


class _NoopThread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


JD = "We need a senior Python engineer with Flask and PostgreSQL experience. " * 2


@pytest.mark.integration
class TestTailorFromSavedResume:
    def test_requires_auth(self, client, db):
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m)
            resp = client.post(
                "/api/tailor",
                data={"saved_resume_id": "whatever", "job_description": JD},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 401

    def test_bogus_id_404(self, client, user, db):
        _login(client, user)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m)
            resp = client.post(
                "/api/tailor",
                data={"saved_resume_id": "doesnotexist", "job_description": JD},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 404

    def test_other_users_resume_404(self, client, user, other_user, db):
        other_resume = _make_saved(db, other_user)
        _login(client, user)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m)
            resp = client.post(
                "/api/tailor",
                data={"saved_resume_id": other_resume.id, "job_description": JD},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 404

    def test_happy_path_writes_saved_text(self, client, user, db, monkeypatch):
        resume = _make_saved(db, user, text="# Jane\nUnique-saved-marker Python Flask")
        _login(client, user)
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m)
            resp = client.post(
                "/api/tailor",
                data={"saved_resume_id": resume.id, "job_description": JD},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        job_id = resp.get_json()["job_id"]
        from app.services.pipeline import jobs

        written = Path(jobs[job_id]["output_dir"]) / "input_resume.txt"
        assert written.exists()
        assert "Unique-saved-marker" in written.read_text(encoding="utf-8")

    def test_no_resume_and_no_saved_id_still_400(self, client, user, db):
        _login(client, user)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m)
            resp = client.post(
                "/api/tailor", data={"job_description": JD}, content_type="multipart/form-data"
            )
        assert resp.status_code == 400
        assert "resume" in resp.get_json()["error"].lower()


@pytest.mark.integration
class TestRenameWithoutText:
    def test_rename_only(self, client, user, db):
        resume = _make_saved(db, user, text="original text", name="Old Name")
        _login(client, user)
        resp = client.post("/api/saved-resumes", json={"id": resume.id, "name": "New Name"})
        assert resp.status_code == 200
        refreshed = db.session.get(SavedResume, resume.id)
        assert refreshed.name == "New Name"
        assert refreshed.resume_text == "original text"  # unchanged

    def test_create_still_requires_text(self, client, user, db):
        _login(client, user)
        resp = client.post("/api/saved-resumes", json={"name": "No Text"})
        assert resp.status_code == 400
