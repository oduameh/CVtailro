"""Integration tests for the per-user daily job budget cap."""

import io
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.models import TailoringJob, User


@pytest.fixture
def user(db):
    u = User(google_id="budget-g-1", email="budget@example.com", name="Budget User", is_admin=False)
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def admin_user(db):
    u = User(google_id="budget-admin-1", email="boss@example.com", name="Boss", is_admin=True)
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user):
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user)


def _config_mock(m, daily_job_limit):
    m.return_value.api_key = "sk-test"
    m.return_value.allow_user_model_selection = False
    m.return_value.default_model = "gpt-4o-mini"
    m.return_value.rate_limit_per_hour = 100
    m.return_value.daily_job_limit = daily_job_limit


def _seed_jobs(db, user_id, count, when=None):
    when = when or datetime.now(timezone.utc)
    for i in range(count):
        db.session.add(
            TailoringJob(id=f"bj{user_id[:6]}{i:04d}", user_id=user_id, status="complete", created_at=when)
        )
    db.session.commit()


def _tailor_payload():
    return {
        "resume": (io.BytesIO(b"# Jane\nPython Flask engineer with experience"), "resume.txt"),
        "job_description": "We need a senior Python engineer with Flask and PostgreSQL skills. " * 2,
    }


@pytest.mark.integration
class TestDailyBudgetCap:
    def test_over_limit_returns_429(self, client, user, db):
        _seed_jobs(db, user.id, 2)
        _login(client, user)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=2)
            resp = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert resp.status_code == 429
        body = resp.get_json()
        assert "daily" in body["error"].lower()
        assert body["daily_limit"] == 2
        assert body["used_today"] == 2

    def test_under_limit_starts_job(self, client, user, db, monkeypatch):
        _seed_jobs(db, user.id, 1)
        _login(client, user)
        # Prevent the real pipeline thread from running.
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=5)
            resp = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert resp.status_code == 200
        assert "job_id" in resp.get_json()

    def test_admin_is_exempt(self, client, admin_user, db, monkeypatch):
        _seed_jobs(db, admin_user.id, 5)
        _login(client, admin_user)
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=2)
            resp = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert resp.status_code != 429

    def test_yesterdays_jobs_dont_count(self, client, user, db, monkeypatch):
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        _seed_jobs(db, user.id, 5, when=yesterday)
        _login(client, user)
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=2)
            resp = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_zero_limit_disables_cap(self, client, user, db, monkeypatch):
        _seed_jobs(db, user.id, 50)
        _login(client, user)
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=0)
            resp = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_anonymous_cap_atomic_check_and_record(self, client, db, monkeypatch):
        # Anonymous users are capped via the in-memory rolling-24h counter
        # (check_and_record_daily). With limit=1 the second request is rejected.
        monkeypatch.setattr("app.routes.api.threading.Thread", _NoopThread)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            _config_mock(m, daily_job_limit=1)
            first = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
            second = client.post("/api/tailor", data=_tailor_payload(), content_type="multipart/form-data")
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.get_json()["daily_limit"] == 1


class _NoopThread:
    """Drop-in for threading.Thread that never actually runs the target."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass
