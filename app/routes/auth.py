"""Authentication routes — Google OAuth only."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import csrf, db, limiter, oauth
from app.models import User
from app.services.telemetry import track

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _get_admin_emails() -> list[str]:
    env = os.environ.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in env.split(",") if e.strip()] if env else []


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "unknown"


def _is_local_request() -> bool:
    ip = _client_ip()
    return ip in {"127.0.0.1", "::1", "localhost"}


# ═══════════════════════════════════════════════════════════════════════════
#  Google OAuth
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

    user = User.query.filter_by(google_id=google_id).first()

    if user is None:
        user = User.query.filter_by(email=email).first()
        if user is not None:
            user.google_id = google_id
            user.email_verified = True
            user.email_verified_at = user.email_verified_at or datetime.now(timezone.utc)
            logger.info(f"Linked Google account to existing user: {email}")

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

    user.is_admin = email in _get_admin_emails()
    user.last_login_at = datetime.now(timezone.utc)
    db.session.commit()
    login_user(user, remember=True)

    # Create tracked session
    from app.services.session_manager import create_session

    create_session(user, request)

    logger.info(f"User logged in via Google: {email} (admin={user.is_admin})")
    track("auth.login.succeeded", category="auth", user_id=user.id, metadata={"provider": "google", "is_admin": user.is_admin})
    return redirect("/")


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


# ═══════════════════════════════════════════════════════════════════════════
#  Session Management (user-facing)
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/sessions")
@login_required
def list_sessions():
    """List all active sessions for the current user."""
    from flask import session as flask_session

    from app.services.session_manager import get_active_sessions

    current_token = flask_session.get("session_token")
    sessions = get_active_sessions(current_user.id)
    result = []
    for s in sessions:
        result.append({
            "id": s.id,
            "ip_address": s.ip_address,
            "device_type": s.device_type,
            "browser_name": s.browser_name,
            "os_name": s.os_name,
            "country": s.country,
            "city": s.city,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "last_activity_at": s.last_activity_at.isoformat() if s.last_activity_at else None,
            "is_current": s.session_token == current_token,
        })
    return jsonify({"sessions": result})


@auth_bp.route("/sessions/<session_id>", methods=["DELETE"])
@login_required
def revoke_session(session_id: str):
    """Revoke a specific session belonging to the current user."""
    from flask import session as flask_session

    from app.services.session_manager import revoke_session as do_revoke

    current_token = flask_session.get("session_token")
    from app.models.user_session import UserSession

    sess = db.session.get(UserSession, session_id)
    if not sess or sess.user_id != current_user.id:
        return jsonify({"error": "Session not found"}), 404
    if sess.session_token == current_token:
        return jsonify({"error": "Cannot revoke your current session. Use logout instead."}), 400

    do_revoke(session_id, reason="manual")
    return jsonify({"ok": True})


@auth_bp.route("/sessions/others", methods=["DELETE"])
@login_required
def revoke_other_sessions():
    """Revoke all sessions except the current one."""
    from flask import session as flask_session

    from app.services.session_manager import revoke_all_other_sessions

    current_token = flask_session.get("session_token")
    count = revoke_all_other_sessions(current_user.id, current_token)
    return jsonify({"ok": True, "revoked_count": count})


@auth_bp.route("/heartbeat", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def heartbeat():
    """Lightweight session keep-alive — extends idle timer."""
    from flask import session as flask_session

    from app.services.session_manager import update_activity

    token = flask_session.get("session_token")
    if token:
        update_activity(token, force=True)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
#  Shared Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@auth_bp.route("/logout", methods=["POST"])
def logout():
    uid = current_user.id if current_user.is_authenticated else None

    from flask import session as flask_session

    from app.services.session_manager import log_login_event, revoke_session_by_token

    token = flask_session.get("session_token")
    if token:
        revoke_session_by_token(token, reason="logout")

    if uid:
        log_login_event(
            user_id=uid,
            event_type="logout",
            ip_address=_client_ip(),
            user_agent=request.headers.get("User-Agent", ""),
        )

    logout_user()
    track("auth.logout", category="auth", user_id=uid)
    return jsonify({"ok": True})


@auth_bp.route("/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})

    from flask import session as flask_session

    from app.services.session_manager import get_session_info

    token = flask_session.get("session_token")
    session_info = get_session_info(token) if token else None

    return jsonify(
        {
            "authenticated": True,
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "picture": current_user.picture_url,
            "is_admin": current_user.is_admin,
            "session": {
                "created_at": session_info.created_at.isoformat() if session_info and session_info.created_at else None,
                "last_activity_at": session_info.last_activity_at.isoformat() if session_info and session_info.last_activity_at else None,
                "expires_at": session_info.expires_at.isoformat() if session_info and session_info.expires_at else None,
                "ip_address": session_info.ip_address if session_info else None,
                "device_type": session_info.device_type if session_info else None,
            } if session_info else None,
        }
    )


@auth_bp.route("/dev-login")
@csrf.exempt
def dev_login():
    from flask import current_app

    if (
        not current_app.debug
        or os.environ.get("FLASK_ENV") == "production"
        or os.environ.get("DEV_AUTH_BYPASS") != "1"
        or not _is_local_request()
    ):
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
    login_user(user, remember=True)

    from app.services.session_manager import create_session

    create_session(user, request)

    return redirect("/")
