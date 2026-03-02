"""Google OAuth authentication routes."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, url_for
from flask_login import current_user, login_user, logout_user

from app.extensions import csrf, db, oauth
from app.models import User

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
csrf.exempt(auth_bp)  # OAuth callbacks don't include CSRF tokens


def _get_admin_emails() -> list[str]:
    env = os.environ.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in env.split(",") if e.strip()] if env else []


@auth_bp.route("/google/login")
def google_login():
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo") or oauth.google.userinfo()
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        return redirect("/?auth_error=oauth_failed")

    google_id = userinfo["sub"]
    email = userinfo.get("email", "").lower()
    name = userinfo.get("name", email.split("@")[0])
    picture = userinfo.get("picture", "")

    if not email:
        return redirect("/?auth_error=no_email")

    user = User.query.filter_by(google_id=google_id).first()
    if user is None:
        user = User(google_id=google_id, email=email, name=name, picture_url=picture)
        db.session.add(user)
        logger.info(f"New user created: {email}")
    else:
        user.name = name
        user.picture_url = picture
        user.email = email

    # Admin: env list takes precedence; otherwise respect DB (allows admin panel promotion)
    user.is_admin = email in _get_admin_emails() or user.is_admin
    user.last_login_at = datetime.now(timezone.utc)
    db.session.commit()
    login_user(user, remember=True)
    logger.info(f"User logged in: {email} (admin={user.is_admin})")
    return redirect("/")


@auth_bp.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return jsonify({"ok": True})


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
        }
    )
