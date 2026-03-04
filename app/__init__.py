"""CVtailro Flask Application Factory.

Usage:
    from app import create_app
    app = create_app()
"""

from __future__ import annotations

import logging
import os

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

from app.extensions import csrf, db, limiter, login_manager, migrate, oauth
from app.routes import register_blueprints
from app.settings import settings_map
from storage import r2_storage

logger = logging.getLogger("cvtailro")


def create_app(config_name: str | None = None) -> Flask:
    """Application factory — creates and configures the Flask app."""

    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "production")

    flask_app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
    )

    settings_cls = settings_map.get(config_name, settings_map["production"])
    flask_app.config.from_object(settings_cls())

    if config_name == "production":
        settings_cls._check_secret_key()

    if config_name != "development" and not flask_app.config.get("TESTING"):
        flask_app.wsgi_app = ProxyFix(  # type: ignore[assignment]
            flask_app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
        )

    from app.middleware import init_request_id, init_sentry, init_structured_logging

    init_structured_logging(flask_app)
    init_sentry(flask_app)
    _init_extensions(flask_app)
    _init_oauth(flask_app)
    _init_redis()
    register_blueprints(flask_app)
    _register_security_headers(flask_app)
    _register_error_handlers(flask_app)
    _register_session_validation(flask_app)
    init_request_id(flask_app)
    _run_migrations(flask_app)
    _register_context_processors(flask_app)

    return flask_app


def _init_extensions(flask_app: Flask) -> None:
    db.init_app(flask_app)
    migrate.init_app(flask_app, db)
    login_manager.init_app(flask_app)
    csrf.init_app(flask_app)
    limiter.init_app(flask_app)

    @login_manager.user_loader
    def load_user(user_id: str):
        from app.models import User

        return db.session.get(User, user_id)

    @login_manager.unauthorized_handler
    def unauthorized():
        from flask import jsonify

        return jsonify({"error": "Authentication required"}), 401

    r2_storage.init_app(flask_app)


def _init_redis() -> None:
    from app.services.cache import init_redis

    init_redis()


def _init_oauth(flask_app: Flask) -> None:
    oauth.init_app(flask_app)
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=os.environ.get("GOOGLE_CLIENT_ID", ""),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        client_kwargs={"scope": "openid email profile"},
    )


def _register_error_handlers(flask_app: Flask) -> None:
    """Return JSON errors for API routes instead of HTML error pages."""
    from flask import jsonify as _jsonify

    @flask_app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
            return _jsonify({"error": "Not found"}), 404
        return e  # Let Flask render default HTML for non-API routes

    @flask_app.errorhandler(500)
    def internal_error(e):
        logger.exception("Unhandled server error")
        if request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
            return _jsonify({"error": "Internal server error"}), 500
        return e

    @flask_app.errorhandler(Exception)
    def handle_exception(e):
        from werkzeug.exceptions import HTTPException

        if isinstance(e, HTTPException):
            if request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
                return _jsonify({"error": e.description}), e.code
            return e
        logger.exception("Unhandled exception in %s %s", request.method, request.path)
        if request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
            return _jsonify({"error": "Internal server error"}), 500
        return e


def _register_security_headers(flask_app: Flask) -> None:
    @flask_app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://accounts.google.com https://apis.google.com "
            "https://pagead2.googlesyndication.com https://www.googletagservices.com "
            "https://tpc.googlesyndication.com https://googleads.g.doubleclick.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https: blob:; "
            "connect-src 'self' https://accounts.google.com "
            "https://pagead2.googlesyndication.com https://googleads.g.doubleclick.net "
            "https://tpc.googlesyndication.com https://www.googletagservices.com; "
            "frame-src https://accounts.google.com https://googleads.g.doubleclick.net "
            "https://tpc.googlesyndication.com https://pagead2.googlesyndication.com; "
            "base-uri 'self'; "
            "form-action 'self' https://accounts.google.com"
        )
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if response.content_type and "text/html" in response.content_type:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Double-submit CSRF cookie for SPA requests.
        # Regenerate when cookie is missing OR when the session lost its
        # CSRF token (e.g. after session backend changes or cookie expiry).
        from flask import session as _sess
        from flask_wtf.csrf import generate_csrf

        if "csrf_token" not in request.cookies or "csrf_token" not in _sess:
            token = generate_csrf()
            response.set_cookie(
                "csrf_token",
                token,
                httponly=False,
                samesite="Lax",
                secure=request.is_secure,
            )
        return response


def _register_session_validation(flask_app: Flask) -> None:
    """Validate server-side session on every authenticated request."""

    _SKIP_PREFIXES = ("/static/", "/auth/google/", "/auth/dev-login", "/favicon.ico")

    @flask_app.before_request
    def validate_session():
        from flask import session as flask_session
        from flask_login import current_user

        if not current_user.is_authenticated:
            return

        if any(request.path.startswith(p) for p in _SKIP_PREFIXES):
            return

        token = flask_session.get("session_token")
        if not token:
            # Legacy session without server-side tracking — force re-login
            from flask_login import logout_user

            logout_user()
            return

        from app.services.session_manager import update_activity
        from app.services.session_manager import validate_session as _validate

        is_valid = _validate(token)
        if not is_valid:
            from flask_login import logout_user

            flask_session.clear()
            logout_user()
            if request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
                from flask import jsonify

                return jsonify({"error": "Session expired. Please sign in again."}), 401
            return

        update_activity(token)


def _register_context_processors(flask_app: Flask) -> None:
    @flask_app.context_processor
    def inject_ads_config():
        from app.services.admin_config import AdminConfigManager

        client_id = AdminConfigManager.get("adsense_client_id") or flask_app.config.get("ADSENSE_CLIENT_ID", "")
        enabled = AdminConfigManager.get("adsense_enabled")
        return {
            "ads_config": {
                "enabled": enabled == "true" and bool(client_id),
                "client_id": client_id,
            }
        }


def _run_migrations(flask_app: Flask) -> None:
    """Create tables and run any pending schema migrations."""
    with flask_app.app_context():
        try:
            db.create_all()
            _apply_column_migrations()
        except Exception as e:
            logger.warning(f"Migration failed (may be first boot): {e}")


def _apply_column_migrations() -> None:
    """Add missing columns/indexes that were added after initial schema."""
    from sqlalchemy import inspect, text

    insp = inspect(db.engine)

    # Ensure admin_settings table exists (added in app factory refactor)
    if not insp.has_table("admin_settings"):
        try:
            db.session.execute(
                text(
                    "CREATE TABLE admin_settings ("
                    "id VARCHAR(32) PRIMARY KEY, "
                    "key VARCHAR(255) UNIQUE NOT NULL, "
                    "value TEXT, "
                    "updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW())"
                )
            )
            db.session.execute(text("CREATE INDEX idx_admin_settings_key ON admin_settings(key)"))
            db.session.commit()
            logger.info("Created admin_settings table")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"Failed to create admin_settings table: {e}")

    if not insp.has_table("tailoring_jobs"):
        return

    columns = [c["name"] for c in insp.get_columns("tailoring_jobs")]

    text_columns = [
        "job_description_snippet",
        "job_description_full",
        "original_resume_text",
        "cover_letter_md",
        "email_templates_md",
    ]
    for col_name in text_columns:
        if col_name not in columns:
            col_type = "VARCHAR(500)" if col_name == "job_description_snippet" else "TEXT"
            db.session.execute(text(f"ALTER TABLE tailoring_jobs ADD COLUMN {col_name} {col_type}"))
            db.session.commit()

    if "original_match_score" not in columns:
        db.session.execute(text("ALTER TABLE tailoring_jobs ADD COLUMN original_match_score FLOAT"))
        db.session.commit()

    json_columns = ["section_scores", "resume_quality_json", "ats_check_json", "keyword_density_json"]
    for col_name in json_columns:
        if col_name not in columns:
            db.session.execute(text(f"ALTER TABLE tailoring_jobs ADD COLUMN {col_name} JSON"))
            db.session.commit()

    existing_indexes = {idx["name"] for idx in insp.get_indexes("tailoring_jobs")}
    for idx_name, idx_sql in [
        (
            "idx_tailoring_jobs_user_created",
            "CREATE INDEX idx_tailoring_jobs_user_created ON tailoring_jobs(user_id, created_at DESC)",
        ),
        (
            "idx_tailoring_jobs_user_status",
            "CREATE INDEX idx_tailoring_jobs_user_status ON tailoring_jobs(user_id, status)",
        ),
    ]:
        if idx_name not in existing_indexes:
            try:
                db.session.execute(text(idx_sql))
                db.session.commit()
            except Exception:
                db.session.rollback()

    # --- Analytics tables ---
    for tbl_name in ("analytics_events", "daily_metrics"):
        if not insp.has_table(tbl_name):
            try:
                db.create_all()
                logger.info(f"Created table {tbl_name}")
            except Exception as e:
                logger.warning(f"Failed to create {tbl_name}: {e}")

    # --- User table: email/password auth columns ---
    if insp.has_table("users"):
        user_columns = [c["name"] for c in insp.get_columns("users")]

        user_new_columns = [
            ("password_hash", "VARCHAR(255)"),
            ("email_verified", "BOOLEAN DEFAULT FALSE"),
            ("email_verified_at", "TIMESTAMP WITH TIME ZONE"),
            ("auth_provider", "VARCHAR(20) DEFAULT 'google'"),
        ]
        for col_name, col_type in user_new_columns:
            if col_name not in user_columns:
                try:
                    db.session.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                    db.session.commit()
                    logger.info(f"Added users.{col_name} column")
                except Exception as e:
                    db.session.rollback()
                    logger.warning(f"Failed to add users.{col_name}: {e}")

        # Make google_id nullable (was NOT NULL for Google-only auth)
        try:
            db.session.execute(text("ALTER TABLE users ALTER COLUMN google_id DROP NOT NULL"))
            db.session.commit()
        except Exception:
            db.session.rollback()  # Already nullable or SQLite (no ALTER support)

    # --- Session tracking tables ---
    for tbl_name in ("user_sessions", "login_events"):
        if not insp.has_table(tbl_name):
            try:
                db.create_all()
                logger.info(f"Created table {tbl_name}")
            except Exception as e:
                logger.warning(f"Failed to create {tbl_name}: {e}")

    if insp.has_table("user_sessions"):
        try:
            us_indexes = {idx["name"] for idx in insp.get_indexes("user_sessions")}
            if "idx_user_sessions_user_active" not in us_indexes:
                db.session.execute(
                    text("CREATE INDEX idx_user_sessions_user_active ON user_sessions(user_id, is_active)")
                )
                db.session.commit()
        except Exception:
            db.session.rollback()

    if insp.has_table("login_events"):
        try:
            le_indexes = {idx["name"] for idx in insp.get_indexes("login_events")}
            if "idx_login_events_user_created" not in le_indexes:
                db.session.execute(
                    text("CREATE INDEX idx_login_events_user_created ON login_events(user_id, created_at DESC)")
                )
                db.session.commit()
        except Exception:
            db.session.rollback()
