"""Integration tests — premium model access control end-to-end."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from app.models import User
from app.services.model_access import is_premium_model

pytestmark = pytest.mark.integration

FREE_ID = "nvidia/nemotron-3-super-120b-a12b:free"
PAID_ID = "openai/gpt-4o-mini"


@pytest.fixture
def user(db):
    u = User(google_id="premium-google-1", email="free@example.com", name="Free User", is_admin=False)
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def granted_user(db):
    u = User(
        google_id="premium-google-2",
        email="granted@example.com",
        name="Granted User",
        is_admin=False,
        can_use_premium_models=True,
    )
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def admin_user(db):
    u = User(google_id="premium-google-3", email="admin@example.com", name="Admin", is_admin=True)
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user_obj):
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user_obj)


def _admin_login(client):
    with client.session_transaction() as sess:
        sess["admin_authenticated"] = True
        sess["admin_auth_at"] = __import__("time").time()


def _tailor_config_mock():
    """Patch AdminConfigManager.load with a permissive tailoring config."""
    patcher = patch("app.services.admin_config.AdminConfigManager.load")
    mock = patcher.start()
    mock.return_value.api_key = "sk-test"
    mock.return_value.allow_user_model_selection = True
    mock.return_value.default_model = FREE_ID
    mock.return_value.rate_limit_per_hour = 100
    mock.return_value.daily_job_limit = 0
    return patcher


def _tailor_post(client, model=None):
    data = {
        "resume": (
            io.BytesIO(b"Senior engineer with 10 years shipping Python backends." * 3),
            "resume.txt",
        ),
        "job_description": "x" * 100,
    }
    if model:
        data["model"] = model
    return client.post("/api/tailor", data=data, content_type="multipart/form-data")


@pytest.fixture(autouse=True)
def _reset_admin_config_cache():
    """Clear AdminConfigManager's 5s TTL class-level cache between tests.

    The session-scoped app keeps the cache alive across test functions, so a
    config loaded by an earlier test would shadow rows inserted here.
    """
    from app.services.admin_config import AdminConfigManager

    def _clear():
        with AdminConfigManager._lock:
            AdminConfigManager._cache = None
            AdminConfigManager._cache_time = 0

    _clear()
    yield
    _clear()


class TestModelsEndpointFiltering:
    def test_anonymous_sees_free_only(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["user_can_use_premium"] is False
        assert data["models"], "expected non-empty catalog"
        assert all(m["tier"] == "free" for m in data["models"])
        assert PAID_ID not in {m["id"] for m in data["models"]}
        assert data["default"]
        assert isinstance(data["user_selectable"], bool)

    def test_free_user_sees_free_only(self, client, user):
        _login(client, user)
        data = client.get("/api/models").get_json()

        assert data["user_can_use_premium"] is False
        assert PAID_ID not in {m["id"] for m in data["models"]}
        assert all(m["tier"] == "free" for m in data["models"])

    def test_granted_user_sees_premium_with_tiers(self, client, granted_user):
        _login(client, granted_user)
        data = client.get("/api/models").get_json()

        assert data["user_can_use_premium"] is True
        tiers = {m["id"]: m["tier"] for m in data["models"]}
        assert tiers.get(PAID_ID) == "premium"
        assert tiers.get(FREE_ID) == "free"
        assert all(tier == ("premium" if is_premium_model(mid) else "free") for mid, tier in tiers.items())

    def test_admin_sees_premium_with_tiers(self, client, admin_user):
        _login(client, admin_user)
        data = client.get("/api/models").get_json()

        assert data["user_can_use_premium"] is True
        assert {m["id"] for m in data["models"] if m["tier"] == "premium"}

    def test_every_model_has_tier_field(self, client, admin_user):
        _login(client, admin_user)
        data = client.get("/api/models").get_json()

        assert data["models"]
        assert all(m["tier"] in ("free", "premium") for m in data["models"])


class TestTailorPremiumGate:
    def test_free_user_with_premium_model_rejected_403(self, client, user):
        _login(client, user)
        patcher = _tailor_config_mock()
        try:
            resp = _tailor_post(client, model=PAID_ID)
        finally:
            patcher.stop()

        assert resp.status_code == 403
        assert resp.get_json()["error"] == "This model requires premium access."

    def test_granted_user_with_premium_model_passes_validation(self, client, granted_user):
        _login(client, granted_user)
        patcher = _tailor_config_mock()
        try:
            with patch("app.services.pipeline.run_pipeline_job"), patch("threading.Thread"):
                resp = _tailor_post(client, model=PAID_ID)
        finally:
            patcher.stop()

        assert resp.status_code == 200
        assert resp.get_json()["job_id"]

    def test_admin_with_premium_model_passes_validation(self, client, admin_user):
        _login(client, admin_user)
        patcher = _tailor_config_mock()
        try:
            with patch("app.services.pipeline.run_pipeline_job"), patch("threading.Thread"):
                resp = _tailor_post(client, model=PAID_ID)
        finally:
            patcher.stop()

        assert resp.status_code == 200
        assert resp.get_json()["job_id"]

    def test_free_user_with_free_model_unaffected(self, client, user):
        _login(client, user)
        patcher = _tailor_config_mock()
        try:
            with patch("app.services.pipeline.run_pipeline_job"), patch("threading.Thread"):
                resp = _tailor_post(client, model=FREE_ID)
        finally:
            patcher.stop()

        assert resp.status_code == 200
        assert resp.get_json()["job_id"]

    def test_anonymous_with_premium_model_rejected_403(self, client):
        patcher = _tailor_config_mock()
        try:
            resp = _tailor_post(client, model=PAID_ID)
        finally:
            patcher.stop()

        assert resp.status_code == 403
        assert resp.get_json()["error"] == "This model requires premium access."


class TestAdminPremiumGrantEndpoint:
    def test_admin_can_toggle_grant_on(self, client, db, user):
        _admin_login(client)
        resp = client.post(
            f"/admin/api/users/{user.id}/premium-models",
            json={"enabled": True},
            content_type="application/json",
        )

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["user_id"] == user.id
        assert body["can_use_premium_models"] is True
        db.session.refresh(user)
        assert user.can_use_premium_models is True

    def test_admin_can_toggle_grant_off(self, client, db, granted_user):
        _admin_login(client)
        resp = client.post(
            f"/admin/api/users/{granted_user.id}/premium-models",
            json={"enabled": False},
            content_type="application/json",
        )

        assert resp.status_code == 200
        assert resp.get_json()["can_use_premium_models"] is False
        db.session.refresh(granted_user)
        assert granted_user.can_use_premium_models is False

    def test_unknown_user_returns_404(self, client):
        _admin_login(client)
        resp = client.post(
            "/admin/api/users/nonexistent123/premium-models",
            json={"enabled": True},
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_non_admin_cannot_grant(self, client, user):
        _login(client, user)
        resp = client.post(
            f"/admin/api/users/{user.id}/premium-models",
            json={"enabled": True},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_anonymous_cannot_grant(self, client, user):
        resp = client.post(
            f"/admin/api/users/{user.id}/premium-models",
            json={"enabled": True},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_users_list_includes_can_use_premium_models(self, client, granted_user):
        _admin_login(client)
        data = client.get("/admin/api/users").get_json()

        row = next(u for u in data["users"] if u["id"] == granted_user.id)
        assert row["can_use_premium_models"] is True


class TestAdminConfigPremiumSetting:
    def test_config_get_returns_premium_models_enabled(self, client, db):
        from app.models.admin_config import AdminSetting

        db.session.add(AdminSetting(key="premium_models_enabled", value="false"))
        db.session.commit()

        _admin_login(client)
        data = client.get("/admin/api/config").get_json()
        assert data["premium_models_enabled"] is False

    def test_config_get_defaults_to_true_when_unset(self, client):
        _admin_login(client)
        data = client.get("/admin/api/config").get_json()
        assert data["premium_models_enabled"] is True

    def test_config_post_persists_premium_models_enabled(self, client, db):
        _admin_login(client)
        resp = client.post(
            "/admin/api/config",
            json={"premium_models_enabled": False},
            content_type="application/json",
        )

        assert resp.status_code == 200
        from app.models.admin_config import AdminSetting

        row = AdminSetting.query.filter_by(key="premium_models_enabled").first()
        assert row is not None
        assert row.value == "false"

        # Round-trip: read back as disabled
        data = client.get("/admin/api/config").get_json()
        assert data["premium_models_enabled"] is False

    def test_kill_switch_hides_premium_from_granted_user(self, client, db, granted_user):
        from app.models.admin_config import AdminSetting

        db.session.add(AdminSetting(key="premium_models_enabled", value="false"))
        db.session.commit()

        _login(client, granted_user)
        data = client.get("/api/models").get_json()

        assert data["user_can_use_premium"] is False
        assert PAID_ID not in {m["id"] for m in data["models"]}
