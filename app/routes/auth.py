"""Authentication routes — Google OAuth + email/password."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf.csrf import generate_csrf
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import csrf, db, limiter, oauth
from app.models import User
from app.services.email import (
    confirm_reset_token,
    confirm_verification_token,
    send_reset_email,
    send_verification_email,
)
from app.services.telemetry import track
from app.services.usage import login_rate_limiter

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_MIN_PASSWORD_LEN = 8
_PW_HASH_METHOD = "pbkdf2:sha256:600000"


def _get_admin_emails() -> list[str]:
    env = os.environ.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in env.split(",") if e.strip()] if env else []


def _validate_email(email: str) -> str | None:
    """Return cleaned email or None if invalid."""
    email = (email or "").strip().lower()
    if not email or not _EMAIL_RE.match(email):
        return None
    return email


def _validate_password(password: str) -> str | None:
    """Return an error message if the password is too weak, else None."""
    if not password or len(password) < _MIN_PASSWORD_LEN:
        return f"Password must be at least {_MIN_PASSWORD_LEN} characters."
    return None


def _client_ip() -> str:
    return request.remote_addr or "unknown"


def _login_and_bind_session(user: User, remember: bool = False) -> None:
    """Log in the user and store session_version for revocation checks."""
    login_user(user, remember=remember)
    session["_session_version"] = user.session_version
    generate_csrf()


# ═══════════════════════════════════════════════════════════════════════════
#  Google OAuth (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/google/login")
@csrf.exempt
def google_login():
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
@csrf.exempt
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo") or oauth.google.userinfo()
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        track("auth.login.failed", category="auth", metadata={"provider": "google", "reason": "oauth_callback_error"})
        return redirect("/?auth_error=oauth_failed")

    google_id = userinfo["sub"]
    email = userinfo.get("email", "").lower()
    name = userinfo.get("name", email.split("@")[0])
    picture = userinfo.get("picture", "")

    if not email:
        return redirect("/?auth_error=no_email")

    # 1) Lookup by google_id first
    user = User.query.filter_by(google_id=google_id).first()

    # 2) If not found, try linking by email (account created via email/password)
    if user is None:
        user = User.query.filter_by(email=email).first()
        if user is not None:
            # Link Google account to existing email user
            user.google_id = google_id
            user.email_verified = True
            user.email_verified_at = user.email_verified_at or datetime.now(timezone.utc)
            logger.info(f"Linked Google account to existing email user: {email}")

    if user is None:
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture_url=picture,
            auth_provider="google",
            email_verified=True,
            email_verified_at=datetime.now(timezone.utc),
        )
        db.session.add(user)
        logger.info(f"New user created via Google: {email}")
    else:
        user.name = name
        user.picture_url = picture
        user.email = email

    # Admin: env list takes precedence; otherwise respect DB (allows admin panel promotion)
    user.is_admin = email in _get_admin_emails() or user.is_admin
    user.last_login_at = datetime.now(timezone.utc)
    user.reset_failed_logins()
    db.session.commit()
    _login_and_bind_session(user, remember=True)
    session["last_oauth_login_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(f"User logged in via Google: {email} (admin={user.is_admin})")
    track("auth.login.succeeded", category="auth", user_id=user.id, metadata={"provider": "google", "is_admin": user.is_admin})
    return redirect("/")


# ═══════════════════════════════════════════════════════════════════════════
#  Email/Password Registration
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = _validate_email(data.get("email", ""))
    password = data.get("password", "")

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email:
        return jsonify({"error": "A valid email address is required."}), 400

    pw_error = _validate_password(password)
    if pw_error:
        return jsonify({"error": pw_error}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"error": "An account with this email already exists."}), 409

    user = User(
        email=email,
        name=name,
        password_hash=generate_password_hash(password, method=_PW_HASH_METHOD),
        auth_provider="email",
        email_verified=False,
    )
    db.session.add(user)
    db.session.commit()

    send_verification_email(email, name)
    logger.info(f"New email registration: {email}")
    track("auth.register.succeeded", category="auth", user_id=user.id, metadata={"provider": "email"})
    return jsonify({"ok": True, "message": "Account created. Please check your email to verify."}), 201


# ═══════════════════════════════════════════════════════════════════════════
#  Email/Password Login
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    ip = _client_ip()
    if login_rate_limiter.is_blocked(ip):
        return jsonify({"error": "Too many failed attempts. Try again in 15 minutes."}), 429

    data = request.get_json(silent=True) or {}
    email = _validate_email(data.get("email", ""))
    password = data.get("password", "")
    remember = bool(data.get("remember_me", False))

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = User.query.filter_by(email=email).first()

    if user is None or not user.password_hash:
        login_rate_limiter.record_failure(ip)
        track("auth.login.failed", category="auth", metadata={"provider": "email", "reason": "invalid_credentials"})
        return jsonify({"error": "Invalid email or password."}), 401

    if user.is_locked:
        track("auth.login.failed", category="auth", user_id=user.id, metadata={"provider": "email", "reason": "account_locked"})
        return jsonify({"error": "Account temporarily locked due to too many failed attempts. Try again later."}), 429

    if not check_password_hash(user.password_hash, password):
        login_rate_limiter.record_failure(ip)
        user.record_failed_login()
        db.session.commit()
        track("auth.login.failed", category="auth", metadata={"provider": "email", "reason": "wrong_password"})
        return jsonify({"error": "Invalid email or password."}), 401

    if not user.email_verified:
        track("auth.login.failed", category="auth", user_id=user.id, metadata={"provider": "email", "reason": "email_not_verified"})
        return jsonify(
            {
                "error": "Please verify your email before signing in.",
                "needs_verification": True,
            }
        ), 403

    login_rate_limiter.reset(ip)
    user.reset_failed_logins()
    user.last_login_at = datetime.now(timezone.utc)
    # Transparent hash upgrade: re-hash with stronger algorithm on successful login
    if not user.password_hash.startswith("pbkdf2:sha256:600000"):
        user.password_hash = generate_password_hash(password, method=_PW_HASH_METHOD)
    db.session.commit()
    _login_and_bind_session(user, remember=remember)
    logger.info(f"User logged in via email: {email}")
    track("auth.login.succeeded", category="auth", user_id=user.id, metadata={"provider": "email"})
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
#  Email Verification
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/verify/<token>")
@csrf.exempt
def verify_email(token: str):
    email = confirm_verification_token(token)
    if email is None:
        return redirect("/?auth_error=invalid_token")

    user = User.query.filter_by(email=email).first()
    if user is None:
        return redirect("/?auth_error=invalid_token")

    if not user.email_verified:
        user.email_verified = True
        user.email_verified_at = datetime.now(timezone.utc)
        user.last_login_at = datetime.now(timezone.utc)
        db.session.commit()

    _login_and_bind_session(user, remember=False)
    logger.info(f"Email verified: {email}")
    return redirect("/?verified=true")


@auth_bp.route("/resend-verification", methods=["POST"])
@limiter.limit("3 per minute")
def resend_verification():
    data = request.get_json(silent=True) or {}
    email = _validate_email(data.get("email", ""))
    if not email:
        return jsonify({"error": "A valid email is required."}), 400

    user = User.query.filter_by(email=email).first()
    if user and not user.email_verified:
        send_verification_email(email, user.name)

    # Always return success to avoid leaking account existence
    return jsonify({"ok": True, "message": "If that email exists, a verification link has been sent."})


# ═══════════════════════════════════════════════════════════════════════════
#  Password Reset
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/forgot-password", methods=["POST"])
@limiter.limit("3 per minute")
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = _validate_email(data.get("email", ""))
    if not email:
        return jsonify({"error": "A valid email is required."}), 400

    user = User.query.filter_by(email=email).first()
    if user and user.password_hash:
        send_reset_email(email, user.name)

    # Always return success to avoid leaking account existence
    return jsonify({"ok": True, "message": "If that email exists, a reset link has been sent."})


@auth_bp.route("/reset-password/<token>")
@csrf.exempt
def reset_password_page(token: str):
    """Redirect to the SPA with the reset token as a query param."""
    email = confirm_reset_token(token)
    if email is None:
        return redirect("/?auth_error=invalid_reset_token")
    return redirect(f"/?reset_token={token}")


@auth_bp.route("/reset-password", methods=["POST"])
@limiter.limit("5 per minute")
def reset_password():
    data = request.get_json(silent=True) or {}
    token = data.get("token", "")
    password = data.get("password", "")

    if not token:
        return jsonify({"error": "Reset token is required."}), 400

    pw_error = _validate_password(password)
    if pw_error:
        return jsonify({"error": pw_error}), 400

    email = confirm_reset_token(token)
    if email is None:
        return jsonify({"error": "Invalid or expired reset link."}), 400

    user = User.query.filter_by(email=email).first()
    if user is None:
        return jsonify({"error": "Invalid or expired reset link."}), 400

    user.password_hash = generate_password_hash(password, method=_PW_HASH_METHOD)
    user.session_version = (user.session_version or 0) + 1
    user.reset_failed_logins()
    if not user.email_verified:
        user.email_verified = True
        user.email_verified_at = datetime.now(timezone.utc)
    db.session.commit()

    logger.info(f"Password reset for: {email}")
    track("auth.password_reset.completed", category="auth", user_id=user.id)
    return jsonify({"ok": True, "message": "Password updated successfully."})


# ═══════════════════════════════════════════════════════════════════════════
#  Profile Management
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/profile/update", methods=["POST"])
@login_required
def profile_update():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400

    current_user.name = name
    db.session.commit()
    return jsonify({"ok": True, "name": current_user.name})


@auth_bp.route("/profile/change-password", methods=["POST"])
@login_required
def change_password():
    data = request.get_json(silent=True) or {}
    current_pw = data.get("current_password", "")
    new_pw = data.get("new_password", "")

    pw_error = _validate_password(new_pw)
    if pw_error:
        return jsonify({"error": pw_error}), 400

    if current_user.password_hash:
        if not current_pw:
            return jsonify({"error": "Current password is required."}), 400
        if not check_password_hash(current_user.password_hash, current_pw):
            return jsonify({"error": "Current password is incorrect."}), 401
    else:
        last_oauth = session.get("last_oauth_login_at")
        if not last_oauth:
            return jsonify({"error": "Please sign in with Google again before setting a password."}), 403
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_oauth)).total_seconds()
        if elapsed > 300:
            return jsonify({"error": "Session expired. Please sign in with Google again before setting a password."}), 403

    current_user.password_hash = generate_password_hash(new_pw, method=_PW_HASH_METHOD)
    current_user.session_version = (current_user.session_version or 0) + 1
    db.session.commit()
    session["_session_version"] = current_user.session_version
    logger.info(f"Password changed for: {current_user.email}")
    return jsonify({"ok": True, "message": "Password updated successfully."})


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/logout", methods=["POST"])
def logout():
    uid = current_user.id if current_user.is_authenticated else None
    logout_user()
    session.clear()
    track("auth.logout", category="auth", user_id=uid)
    resp = jsonify({"ok": True})
    resp.delete_cookie("remember_token")
    resp.delete_cookie("session")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@auth_bp.route("/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})
    return jsonify(
        {
            "authenticated": True,
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "picture": current_user.picture_url,
            "is_admin": current_user.is_admin,
            "auth_provider": getattr(current_user, "auth_provider", "google"),
            "email_verified": getattr(current_user, "email_verified", True),
            "has_password": current_user.has_password,
        }
    )


@auth_bp.route("/dev-login")
@csrf.exempt
def dev_login():
    from flask import current_app

    if not current_app.debug or os.environ.get("DEV_AUTH_BYPASS") != "1":
        return redirect("/")

    email = os.environ.get("DEV_AUTH_EMAIL", "dev@example.com").lower()
    name = os.environ.get("DEV_AUTH_NAME", "Dev User").strip() or "Dev User"
    google_id = f"dev-{email}"
    user = User.query.filter_by(email=email).first()
    if user is None:
        user = User(google_id=google_id, email=email, name=name, picture_url="")
        db.session.add(user)
    else:
        user.google_id = user.google_id or google_id
        user.name = name
        user.email = email

    if os.environ.get("DEV_AUTH_ADMIN") == "1":
        user.is_admin = True

    user.last_login_at = datetime.now(timezone.utc)
    db.session.commit()
    _login_and_bind_session(user, remember=True)
    return redirect("/")
