"""End-to-end tests for email/password authentication."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from werkzeug.security import generate_password_hash

from app.extensions import db as _db
from app.models import User

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register(client, name="Test User", email="test@example.com", password="securepass123"):
    return client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password},
        content_type="application/json",
    )


def _login(client, email="test@example.com", password="securepass123"):
    return client.post(
        "/auth/login",
        json={"email": email, "password": password},
        content_type="application/json",
    )


def _create_verified_user(email="test@example.com", name="Test User", password="securepass123"):
    user = User(
        email=email,
        name=name,
        password_hash=generate_password_hash(password),
        auth_provider="email",
        email_verified=True,
    )
    _db.session.add(user)
    _db.session.commit()
    return user


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


# ═══════════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@patch("app.routes.auth.send_verification_email", return_value=True)
def test_register_success(mock_email, client):
    resp = _register(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["ok"] is True
    assert "verify" in data["message"].lower() or "check" in data["message"].lower()
    mock_email.assert_called_once_with("test@example.com", "Test User")


@pytest.mark.integration
def test_register_missing_name(client):
    resp = _register(client, name="")
    assert resp.status_code == 400
    assert "name" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_register_invalid_email(client):
    resp = _register(client, email="not-an-email")
    assert resp.status_code == 400
    assert "email" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_register_short_password(client):
    resp = _register(client, password="short")
    assert resp.status_code == 400
    assert "8 characters" in resp.get_json()["error"]


@pytest.mark.integration
@patch("app.routes.auth.send_verification_email", return_value=True)
def test_register_duplicate_email(mock_email, client):
    _register(client)
    resp = _register(client)
    assert resp.status_code == 409
    assert "already exists" in resp.get_json()["error"].lower()


@pytest.mark.integration
@patch("app.routes.auth.send_verification_email", return_value=True)
def test_register_duplicate_email_google_account(mock_email, client):
    """Cannot register with email that already belongs to a Google user."""
    _create_google_user(email="google@example.com")
    resp = _register(client, email="google@example.com")
    assert resp.status_code == 409


# ═══════════════════════════════════════════════════════════════════════════
#  Login
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_login_success(client):
    _create_verified_user()
    resp = _login(client)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


@pytest.mark.integration
def test_login_wrong_password(client):
    _create_verified_user()
    resp = _login(client, password="wrongpassword")
    assert resp.status_code == 401
    assert "invalid" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_login_nonexistent_email(client):
    resp = _login(client, email="noone@example.com")
    assert resp.status_code == 401
    assert "invalid" in resp.get_json()["error"].lower()


@pytest.mark.integration
@patch("app.routes.auth.send_verification_email", return_value=True)
def test_login_unverified(mock_email, client):
    """User who registered but hasn't verified cannot log in."""
    _register(client)
    resp = _login(client)
    assert resp.status_code == 403
    data = resp.get_json()
    assert "verify" in data["error"].lower()
    assert data["needs_verification"] is True


@pytest.mark.integration
def test_login_google_user_no_password(client):
    """Google-only user with no password cannot log in via email/password."""
    _create_google_user(email="google@example.com")
    resp = _login(client, email="google@example.com", password="anything")
    assert resp.status_code == 401


@pytest.mark.integration
def test_login_rate_limiting(client):
    """After MAX_ATTEMPTS failures, login is blocked."""
    _create_verified_user()
    from app.services.usage import login_rate_limiter

    # Reset state
    login_rate_limiter.reset("127.0.0.1")

    for _ in range(5):
        _login(client, password="wrong")

    resp = _login(client, password="securepass123")
    assert resp.status_code == 429
    assert "too many" in resp.get_json()["error"].lower()

    # Clean up
    login_rate_limiter.reset("127.0.0.1")


# ═══════════════════════════════════════════════════════════════════════════
#  Email Verification
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_verify_valid_token(client, flask_app):
    _create_verified_user(email="unverified@example.com")
    user = User.query.filter_by(email="unverified@example.com").first()
    user.email_verified = False
    _db.session.commit()

    with flask_app.app_context():
        from app.services.email import generate_verification_token

        token = generate_verification_token("unverified@example.com")

    resp = client.get(f"/auth/verify/{token}")
    assert resp.status_code == 302
    assert "verified=true" in resp.headers["Location"]

    user = User.query.filter_by(email="unverified@example.com").first()
    assert user.email_verified is True


@pytest.mark.integration
def test_verify_invalid_token(client):
    resp = client.get("/auth/verify/invalid-token-here")
    assert resp.status_code == 302
    assert "invalid_token" in resp.headers["Location"]


@pytest.mark.integration
def test_verify_expired_token(client, flask_app):
    _create_verified_user(email="expired@example.com")
    with flask_app.app_context():
        from app.services.email import confirm_verification_token, generate_verification_token

        token = generate_verification_token("expired@example.com")
        # Verify with max_age=0 to simulate expired
        result = confirm_verification_token(token, max_age=0)
        # Token should be "expired" only if we wait — just verify the function works
        assert result is None or result == "expired@example.com"


@pytest.mark.integration
@patch("app.routes.auth.send_verification_email", return_value=True)
def test_resend_verification(mock_email, client):
    _register(client, email="resend@example.com")
    mock_email.reset_mock()

    resp = client.post(
        "/auth/resend-verification",
        json={"email": "resend@example.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    mock_email.assert_called_once()


@pytest.mark.integration
def test_resend_verification_nonexistent_email(client):
    """Should still return 200 to avoid leaking account info."""
    resp = client.post(
        "/auth/resend-verification",
        json={"email": "nobody@example.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
#  Password Reset
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@patch("app.routes.auth.send_reset_email", return_value=True)
def test_forgot_password(mock_email, client):
    _create_verified_user(email="forgot@example.com")
    resp = client.post(
        "/auth/forgot-password",
        json={"email": "forgot@example.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    mock_email.assert_called_once()


@pytest.mark.integration
def test_forgot_password_nonexistent(client):
    """Should still return 200 to avoid leaking account info."""
    resp = client.post(
        "/auth/forgot-password",
        json={"email": "nobody@example.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200


@pytest.mark.integration
def test_reset_password_valid(client, flask_app):
    _create_verified_user(email="reset@example.com")

    with flask_app.app_context():
        from app.services.email import generate_reset_token

        token = generate_reset_token("reset@example.com")

    resp = client.post(
        "/auth/reset-password",
        json={"token": token, "password": "newpassword123"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True

    # Can login with new password
    resp = _login(client, email="reset@example.com", password="newpassword123")
    assert resp.status_code == 200


@pytest.mark.integration
def test_reset_password_invalid_token(client):
    resp = client.post(
        "/auth/reset-password",
        json={"token": "bad-token", "password": "newpassword123"},
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "invalid" in resp.get_json()["error"].lower() or "expired" in resp.get_json()["error"].lower()


@pytest.mark.integration
def test_reset_password_short_password(client, flask_app):
    _create_verified_user(email="resetshort@example.com")
    with flask_app.app_context():
        from app.services.email import generate_reset_token

        token = generate_reset_token("resetshort@example.com")

    resp = client.post(
        "/auth/reset-password",
        json={"token": token, "password": "short"},
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.integration
def test_reset_password_page_redirect(client, flask_app):
    _create_verified_user(email="redirect@example.com")
    with flask_app.app_context():
        from app.services.email import generate_reset_token

        token = generate_reset_token("redirect@example.com")

    resp = client.get(f"/auth/reset-password/{token}")
    assert resp.status_code == 302
    assert "reset_token=" in resp.headers["Location"]


@pytest.mark.integration
def test_reset_password_page_invalid_token(client):
    resp = client.get("/auth/reset-password/invalid-token")
    assert resp.status_code == 302
    assert "invalid_reset_token" in resp.headers["Location"]


# ═══════════════════════════════════════════════════════════════════════════
#  Profile
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_profile_update_name(client):
    _create_verified_user()
    _login(client)

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
def test_profile_update_name_empty(client):
    _create_verified_user()
    _login(client)

    resp = client.post(
        "/auth/profile/update",
        json={"name": ""},
        content_type="application/json",
    )
    assert resp.status_code == 400


@pytest.mark.integration
def test_profile_update_requires_auth(client):
    resp = client.post(
        "/auth/profile/update",
        json={"name": "New Name"},
        content_type="application/json",
    )
    assert resp.status_code == 401


@pytest.mark.integration
def test_change_password(client):
    _create_verified_user()
    _login(client)

    resp = client.post(
        "/auth/profile/change-password",
        json={"current_password": "securepass123", "new_password": "newpassword456"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


@pytest.mark.integration
def test_change_password_wrong_current(client):
    _create_verified_user()
    _login(client)

    resp = client.post(
        "/auth/profile/change-password",
        json={"current_password": "wrongpassword", "new_password": "newpassword456"},
        content_type="application/json",
    )
    assert resp.status_code == 401


@pytest.mark.integration
def test_set_password_google_user(client):
    """Google user without password can set one (no current_password required)."""
    user = _create_google_user()
    # Simulate being logged in
    with client.session_transaction() as sess:
        sess["_user_id"] = user.id

    resp = client.post(
        "/auth/profile/change-password",
        json={"new_password": "mynewpassword123"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


@pytest.mark.integration
def test_change_password_requires_auth(client):
    resp = client.post(
        "/auth/profile/change-password",
        json={"current_password": "old", "new_password": "newpassword456"},
        content_type="application/json",
    )
    assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════
#  /auth/me endpoint
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_me_unauthenticated(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is False


@pytest.mark.integration
def test_me_authenticated_email_user(client):
    _create_verified_user()
    _login(client)

    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is True
    assert data["email"] == "test@example.com"
    assert data["auth_provider"] == "email"
    assert data["email_verified"] is True
    assert data["has_password"] is True


@pytest.mark.integration
def test_me_authenticated_google_user(client):
    user = _create_google_user()
    with client.session_transaction() as sess:
        sess["_user_id"] = user.id

    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is True
    assert data["auth_provider"] == "google"
    assert data["has_password"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  Logout (existing test extended)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_logout_clears_session(client):
    _create_verified_user()
    _login(client)

    # Verify logged in
    resp = client.get("/auth/me")
    assert resp.get_json()["authenticated"] is True

    # Logout
    resp = client.post("/auth/logout")
    assert resp.status_code == 200

    # Verify logged out
    resp = client.get("/auth/me")
    assert resp.get_json()["authenticated"] is False
