"""
CVtailro Google OAuth Authentication

Provides Google OAuth login/logout, user session management via Flask-Login,
and an admin_required decorator for protected routes.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import wraps

from authlib.integrations.flask_client import OAuth
from flask import Blueprint, jsonify, redirect, session, url_for
from flask_login import LoginManager, current_user, login_user, logout_user

from database import User, db

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
login_manager = LoginManager()
oauth = OAuth()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)


@login_manager.unauthorized_handler
def unauthorized():
    """Return JSON 401 for API requests instead of redirecting."""
    return jsonify({"error": "Authentication required"}), 401


def init_oauth(app):
    """Initialize OAuth with the Flask app and register the Google provider."""
    oauth.init_app(app)
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        client_kwargs={"scope": "openid email profile"},
    )


def _get_admin_emails():
    """Return list of admin email addresses from ADMIN_EMAILS env var."""
    env = os.environ.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in env.split(",") if e.strip()] if env else []


def admin_required(f):
    """Decorator requiring authenticated admin user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401
        if not current_user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated


@auth_bp.route("/google/login")
def google_login():
    """Initiate Google OAuth login flow."""
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/google/callback")
def google_callback():
    """Handle the OAuth callback from Google."""
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
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture_url=picture,
        )
        db.session.add(user)
        logger.info(f"New user created: {email}")
    else:
        user.name = name
        user.picture_url = picture
        user.email = email

    user.is_admin = email in _get_admin_emails()
    user.last_login_at = datetime.now(timezone.utc)
    db.session.commit()
    login_user(user, remember=True)
    logger.info(f"User logged in: {email} (admin={user.is_admin})")
    return redirect("/")


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """Log out the current user."""
    logout_user()
    return jsonify({"ok": True})


@auth_bp.route("/me")
def me():
    """Return current user info, or authenticated=False if not logged in."""
    if not current_user.is_authenticated:
        return jsonify({"authenticated": False})
    return jsonify({
        "authenticated": True,
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "picture": current_user.picture_url,
        "is_admin": current_user.is_admin,
    })
