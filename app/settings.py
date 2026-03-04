"""Flask application settings — one class per environment."""

import os
from datetime import timedelta


def _database_uri() -> str:
    """Normalize DATABASE_URL for SQLAlchemy (postgres:// -> postgresql://)."""
    url = os.environ.get("DATABASE_URL", "sqlite:///cvtailro_dev.db")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _engine_options() -> dict:
    """Connection pool settings for PostgreSQL (no-op for SQLite)."""
    if _database_uri().startswith("sqlite"):
        return {}
    return {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 10,
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }


class BaseSettings:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "cvtailro-dev-secret-change-in-production")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
    PREFERRED_URL_SCHEME = "https"

    # Server-side sessions (Flask-Session backed by SQLAlchemy)
    SESSION_TYPE = "sqlalchemy"
    SESSION_PERMANENT = True
    SESSION_USE_SIGNER = True
    SESSION_SQLALCHEMY_TABLE = "flask_sessions"

    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours

    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_ENGINE_OPTIONS = _engine_options()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour
    WTF_CSRF_HEADERS = ["X-CSRFToken"]
    WTF_CSRF_CHECK_DEFAULT = True

    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = True
    REMEMBER_COOKIE_DURATION = timedelta(days=14)

    # Blog + AdSense
    ADSENSE_CLIENT_ID = os.environ.get("ADSENSE_CLIENT_ID", "")
    ADSENSE_SLOT_TOP = os.environ.get("ADSENSE_SLOT_TOP", "")
    ADSENSE_SLOT_MID = os.environ.get("ADSENSE_SLOT_MID", "")
    ADSENSE_SLOT_BOTTOM = os.environ.get("ADSENSE_SLOT_BOTTOM", "")
    BLOG_BASE_URL = os.environ.get("BLOG_BASE_URL", "https://cvtailro-production.up.railway.app")

    # Pipeline concurrency is configured in app/services/pipeline.py (module-level constants).


class DevelopmentSettings(BaseSettings):
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False


class ProductionSettings(BaseSettings):
    DEBUG = False
    SESSION_COOKIE_SECURE = True

    @classmethod
    def _check_secret_key(cls):
        if not os.environ.get("FLASK_SECRET_KEY"):
            raise RuntimeError(
                "FLASK_SECRET_KEY must be set in production. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )


class TestingSettings(BaseSettings):
    TESTING = True
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}  # type: ignore[assignment]
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


settings_map = {
    "development": DevelopmentSettings,
    "production": ProductionSettings,
    "testing": TestingSettings,
}
