"""Tests for the observability hub: telemetry, PII redaction, analytics models, admin endpoints."""

import time
from datetime import datetime, timedelta, timezone

from app.models import AnalyticsEvent, DailyMetric, TailoringJob, User
from app.models.analytics import hash_user_id
from app.services.telemetry import _redact_value, _sanitize_metadata


def _wait_for_event(event_name: str, timeout: float = 2.0, interval: float = 0.1) -> AnalyticsEvent | None:
    """Poll the DB for an analytics event instead of using a fixed sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        evt = AnalyticsEvent.query.filter_by(event_name=event_name).first()
        if evt is not None:
            return evt
        time.sleep(interval)
    return AnalyticsEvent.query.filter_by(event_name=event_name).first()

# ═══════════════════════════════════════════════════════════════════════════════
# PII Redaction
# ═══════════════════════════════════════════════════════════════════════════════


class TestPIIRedaction:
    def test_redacts_email_key(self):
        assert _redact_value("email", "user@example.com") == "[REDACTED:16chars]"

    def test_redacts_password(self):
        assert _redact_value("password", "s3cret") == "[REDACTED:6chars]"

    def test_redacts_api_key(self):
        result = _redact_value("api_key", "sk-abc123")
        assert result.startswith("[REDACTED:") and result.endswith("chars]")

    def test_redacts_resume_text(self):
        assert _redact_value("resume_text", "John Doe, 10 years experience...") == "[REDACTED:32chars]"

    def test_redacts_job_description(self):
        assert "[REDACTED:" in _redact_value("job_description", "We are looking for...")

    def test_passes_safe_key(self):
        assert _redact_value("model", "gpt-4o-mini") == "gpt-4o-mini"

    def test_redacts_emails_in_string_values(self):
        val = _redact_value("note", "Contact user@example.com for details")
        assert "user@example.com" not in val
        assert "[EMAIL_REDACTED]" in val

    def test_sanitize_metadata_full(self):
        data = {
            "email": "admin@test.com",
            "model": "gpt-4o-mini",
            "password": "secret",
            "resume_text": "My resume content",
            "duration_s": 12.5,
        }
        result = _sanitize_metadata(data)
        assert result["model"] == "gpt-4o-mini"
        assert result["duration_s"] == 12.5
        assert "[REDACTED:" in result["email"]
        assert "[REDACTED:" in result["password"]
        assert "[REDACTED:" in result["resume_text"]

    def test_sanitize_metadata_none(self):
        assert _sanitize_metadata(None) is None

    def test_sanitize_metadata_empty(self):
        assert _sanitize_metadata({}) == {}


# ═══════════════════════════════════════════════════════════════════════════════
# User ID Hashing
# ═══════════════════════════════════════════════════════════════════════════════


class TestUserIdHashing:
    def test_hash_returns_16_chars(self):
        h = hash_user_id("abc123")
        assert len(h) == 16
        assert h.isalnum()

    def test_hash_deterministic(self):
        assert hash_user_id("abc123") == hash_user_id("abc123")

    def test_hash_different_inputs(self):
        assert hash_user_id("user1") != hash_user_id("user2")

    def test_hash_none_returns_none(self):
        assert hash_user_id(None) is None

    def test_hash_empty_returns_none(self):
        assert hash_user_id("") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics Models
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyticsModels:
    def test_create_analytics_event(self, db):
        evt = AnalyticsEvent(
            event_name="test.event",
            category="test",
            user_id_hash="abcdef1234567890",
            request_id="rid123",
            job_id="job456",
            metadata_json={"key": "value"},
        )
        db.session.add(evt)
        db.session.commit()

        fetched = AnalyticsEvent.query.filter_by(event_name="test.event").first()
        assert fetched is not None
        assert fetched.category == "test"
        assert fetched.metadata_json["key"] == "value"
        assert fetched.event_time is not None

    def test_create_daily_metric(self, db):
        from datetime import date

        metric = DailyMetric(
            date=date.today(),
            metric_name="jobs_completed",
            metric_value=42.0,
            dimension_key="model:gpt-4o-mini",
            dimensions={"model": "gpt-4o-mini"},
        )
        db.session.add(metric)
        db.session.commit()

        fetched = DailyMetric.query.filter_by(metric_name="jobs_completed").first()
        assert fetched is not None
        assert fetched.metric_value == 42.0

    def test_analytics_event_indexes(self, db):
        for i in range(5):
            db.session.add(AnalyticsEvent(
                event_name=f"test.event.{i % 2}",
                category="test" if i % 2 == 0 else "other",
            ))
        db.session.commit()

        test_events = AnalyticsEvent.query.filter_by(category="test").count()
        assert test_events == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry Track Function
# ═══════════════════════════════════════════════════════════════════════════════


class TestTelemetryTrack:
    def test_track_persists_event(self, flask_app, db):
        from app.services.telemetry import track_with_app
        with flask_app.app_context():
            track_with_app(
                flask_app, "test.track",
                category="test",
                user_id="user123",
                job_id="job456",
                metadata={"model": "gpt-4o-mini", "duration_s": 5.0},
            )

            evt = _wait_for_event("test.track")
            assert evt is not None
            assert evt.category == "test"
            assert evt.job_id == "job456"
            assert evt.user_id_hash is not None
            assert evt.user_id_hash != "user123"

    def test_track_redacts_pii(self, flask_app, db):
        from app.services.telemetry import track_with_app
        with flask_app.app_context():
            track_with_app(
                flask_app, "test.pii",
                category="test",
                metadata={"email": "secret@test.com", "model": "gpt-4o"},
            )

            evt = _wait_for_event("test.pii")
            assert evt is not None
            meta = evt.metadata_json
            assert "secret@test.com" not in str(meta)
            assert meta["model"] == "gpt-4o"

    def test_track_handles_no_metadata(self, flask_app, db):
        from app.services.telemetry import track_with_app
        with flask_app.app_context():
            track_with_app(flask_app, "test.noop", category="test")

            evt = _wait_for_event("test.noop")
            assert evt is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Admin Observability Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


def _make_admin_user(db):
    u = User(
        email="admin@test.com",
        name="Admin",
        is_admin=True,
        google_id="g-admin",
        auth_provider="google",
        email_verified=True,
    )
    db.session.add(u)
    db.session.commit()
    return u


def _login_admin(client, flask_app, db):
    user = _make_admin_user(db)
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user)
    with client.session_transaction() as sess:
        sess["admin_authenticated"] = True
    return user


def _seed_jobs(db, user, count=5, status="complete"):
    now = datetime.now(timezone.utc)
    for i in range(count):
        j = TailoringJob(
            user_id=user.id,
            status=status,
            model_used="openai/gpt-4o-mini",
            job_title=f"Job {i}",
            company=f"Company {i}",
            duration_seconds=10.0 + i,
            created_at=now - timedelta(days=i),
            completed_at=now - timedelta(days=i) + timedelta(seconds=10+i),
        )
        db.session.add(j)
    db.session.commit()


class TestProductUsageEndpoint:
    def test_returns_dau_wau_mau(self, client, flask_app, db):
        user = _login_admin(client, flask_app, db)
        _seed_jobs(db, user, count=3)

        resp = client.get("/admin/api/observability/product-usage")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "dau" in data
        assert "wau" in data
        assert "mau" in data
        assert "feature_adoption" in data
        assert "jobs_per_day" in data

    def test_feature_adoption_metrics(self, client, flask_app, db):
        _login_admin(client, flask_app, db)
        resp = client.get("/admin/api/observability/product-usage")
        data = resp.get_json()
        fa = data["feature_adoption"]
        assert "saved_resumes_pct" in fa
        assert "tracker_pct" in fa


class TestReliabilityEndpoint:
    def test_returns_reliability_metrics(self, client, flask_app, db):
        user = _login_admin(client, flask_app, db)
        _seed_jobs(db, user, count=4, status="complete")
        _seed_jobs(db, user, count=1, status="error")

        resp = client.get("/admin/api/observability/reliability")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "success_rate" in data
        assert "duration_p50" in data
        assert "duration_p95" in data
        assert "errors_by_model" in data
        assert "live" in data
        assert data["success_rate"] == 80.0

    def test_returns_recent_errors(self, client, flask_app, db):
        user = _login_admin(client, flask_app, db)
        j = TailoringJob(
            user_id=user.id, status="error", model_used="test/model",
            error_message="Test failure", created_at=datetime.now(timezone.utc),
        )
        db.session.add(j)
        db.session.commit()

        resp = client.get("/admin/api/observability/reliability")
        data = resp.get_json()
        assert len(data["recent_errors"]) >= 1
        assert "Test failure" in data["recent_errors"][0]["error"]


class TestCostEndpoint:
    def test_returns_cost_data(self, client, flask_app, db):
        user = _login_admin(client, flask_app, db)
        _seed_jobs(db, user, count=3)

        resp = client.get("/admin/api/observability/cost")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "cost_30d" in data
        assert "model_usage_all" in data
        assert "live_stats" in data


class TestAuditEndpoint:
    def test_returns_audit_trail(self, client, flask_app, db):
        _login_admin(client, flask_app, db)

        with flask_app.app_context():
            db.session.add(AnalyticsEvent(
                event_name="admin.config.updated",
                category="admin",
                metadata_json={"changed_keys": ["default_model"]},
            ))
            db.session.add(AnalyticsEvent(
                event_name="auth.login.failed",
                category="auth",
                metadata_json={"provider": "email", "reason": "wrong_password"},
            ))
            db.session.commit()

        resp = client.get("/admin/api/observability/audit")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "admin_actions" in data
        assert "auth_events" in data
        assert len(data["admin_actions"]) >= 1
        assert len(data["auth_events"]) >= 1


class TestEventFeedEndpoint:
    def test_returns_paginated_events(self, client, flask_app, db):
        _login_admin(client, flask_app, db)

        with flask_app.app_context():
            for i in range(15):
                db.session.add(AnalyticsEvent(
                    event_name=f"test.event.{i}",
                    category="test" if i % 2 == 0 else "auth",
                ))
            db.session.commit()

        resp = client.get("/admin/api/observability/events?per_page=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 15
        assert len(data["events"]) == 10
        assert data["page"] == 1

    def test_filters_by_category(self, client, flask_app, db):
        _login_admin(client, flask_app, db)

        with flask_app.app_context():
            db.session.add(AnalyticsEvent(event_name="auth.login", category="auth"))
            db.session.add(AnalyticsEvent(event_name="tailor.started", category="tailor"))
            db.session.commit()

        resp = client.get("/admin/api/observability/events?category=auth")
        data = resp.get_json()
        assert all(e["category"] == "auth" for e in data["events"])


class TestAlertsEndpoint:
    def test_returns_alerts_and_slo(self, client, flask_app, db):
        _login_admin(client, flask_app, db)

        resp = client.get("/admin/api/observability/alerts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "alerts" in data
        assert "slo" in data
        assert "error_rate_24h" in data["slo"]

    def test_detects_error_rate_spike(self, client, flask_app, db):
        user = _login_admin(client, flask_app, db)
        now = datetime.now(timezone.utc)
        for _i in range(4):
            db.session.add(TailoringJob(
                user_id=user.id, status="error", model_used="test/model",
                error_message="fail", created_at=now - timedelta(hours=1),
            ))
        db.session.add(TailoringJob(
            user_id=user.id, status="complete", model_used="test/model",
            created_at=now - timedelta(hours=1),
        ))
        db.session.commit()

        resp = client.get("/admin/api/observability/alerts")
        data = resp.get_json()
        alert_types = [a["type"] for a in data["alerts"]]
        assert "error_rate_spike" in alert_types


class TestRetentionEndpoint:
    def test_prunes_old_events(self, client, flask_app, db):
        _login_admin(client, flask_app, db)

        with flask_app.app_context():
            old_time = datetime.now(timezone.utc) - timedelta(days=100)
            db.session.add(AnalyticsEvent(
                event_name="old.event", category="test", event_time=old_time,
            ))
            db.session.add(AnalyticsEvent(
                event_name="new.event", category="test",
            ))
            db.session.commit()

        resp = client.post(
            "/admin/api/observability/retention",
            json={"days": 90},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["deleted"] == 1

        remaining = AnalyticsEvent.query.count()
        assert remaining >= 1


class TestObservabilityRequiresAuth:
    def test_product_usage_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/product-usage")
        assert resp.status_code == 401

    def test_reliability_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/reliability")
        assert resp.status_code == 401

    def test_cost_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/cost")
        assert resp.status_code == 401

    def test_audit_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/audit")
        assert resp.status_code == 401

    def test_events_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/events")
        assert resp.status_code == 401

    def test_alerts_requires_admin(self, client, flask_app, db):
        resp = client.get("/admin/api/observability/alerts")
        assert resp.status_code == 401
