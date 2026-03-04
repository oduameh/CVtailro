"""Integration tests for authentication endpoints (Google-only)."""

import pytest

from app.extensions import db as _db
from app.models import User
from tests.conftest import login_user_with_session


def _create_google_user(email="google@example.com", name="Google User"):
    user = User(
        email=email,
        name=name,
        google_id="google-id-12345",
        auth_provider="google",
        email_verified=True,
    )
    _db.session.add(user)
    _db.session.commit()
    return user


@pytest.mark.integration
def test_auth_me_unauthenticated(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is False


@pytest.mark.integration
def test_logout_unauthenticated(client):
    resp = client.post("/auth/logout")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


@pytest.mark.integration
def test_history_requires_auth(client):
    resp = client.get("/api/history")
    assert resp.status_code == 401


@pytest.mark.integration
def test_me_authenticated_google_user(client):
    user = _create_google_user()
    login_user_with_session(client, user)

    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is True
    assert data["email"] == "google@example.com"
    assert data.get("session") is not None


@pytest.mark.integration
def test_logout_clears_session(client):
    user = _create_google_user()
    login_user_with_session(client, user)

    resp = client.get("/auth/me")
    assert resp.get_json()["authenticated"] is True

    resp = client.post("/auth/logout")
    assert resp.status_code == 200

    resp = client.get("/auth/me")
    assert resp.get_json()["authenticated"] is False


@pytest.mark.integration
def test_profile_update_name(client):
    user = _create_google_user()
    login_user_with_session(client, user)

    resp = client.post(
        "/auth/profile/update",
        json={"name": "Updated Name"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["name"] == "Updated Name"


@pytest.mark.integration
def test_profile_update_requires_auth(client):
    resp = client.post(
        "/auth/profile/update",
        json={"name": "New Name"},
        content_type="application/json",
    )
    assert resp.status_code == 401


@pytest.mark.integration
def test_email_login_route_removed(client):
    """Email/password login route should no longer exist."""
    resp = client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "password"},
        content_type="application/json",
    )
    assert resp.status_code in (404, 405)


@pytest.mark.integration
def test_register_route_removed(client):
    """Email/password registration route should no longer exist."""
    resp = client.post(
        "/auth/register",
        json={"name": "Test", "email": "test@example.com", "password": "password123"},
        content_type="application/json",
    )
    assert resp.status_code in (404, 405)


@pytest.mark.integration
def test_sessions_list(client):
    """Authenticated user can list their active sessions."""
    user = _create_google_user()
    login_user_with_session(client, user)

    resp = client.get("/auth/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 1
    assert data["sessions"][0]["is_current"] is True


@pytest.mark.integration
def test_sessions_list_requires_auth(client):
    resp = client.get("/auth/sessions")
    assert resp.status_code == 401


@pytest.mark.integration
def test_heartbeat(client):
    user = _create_google_user()
    login_user_with_session(client, user)

    resp = client.post("/auth/heartbeat")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


@pytest.mark.integration
def test_revoke_other_sessions(client):
    """User can revoke all other sessions."""
    user = _create_google_user()
    login_user_with_session(client, user)

    # Create a second session manually
    from datetime import datetime, timedelta, timezone

    from app.models.user_session import UserSession

    now = datetime.now(timezone.utc)
    other_sess = UserSession(
        user_id=user.id,
        session_token="other-token-12345",
        ip_address="1.2.3.4",
        created_at=now,
        last_activity_at=now,
        expires_at=now + timedelta(hours=24),
        is_active=True,
    )
    _db.session.add(other_sess)
    _db.session.commit()

    resp = client.delete("/auth/sessions/others")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["revoked_count"] == 1

    # Current session should still be active
    resp = client.get("/auth/sessions")
    assert len(resp.get_json()["sessions"]) == 1
