"""Flask application settings — one class per environment."""

import os


class BaseSettings:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "cvtailro-dev-secret-change-in-production")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
    PREFERRED_URL_SCHEME = "https"

    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour

    # Pipeline concurrency
    MAX_CONCURRENT_PIPELINES = 5
    MAX_QUEUE_DEPTH = 50
    JOB_TTL = 900  # 15 minutes

    @staticmethod
    def _database_uri() -> str:
        url = os.environ.get("DATABASE_URL", "sqlite:///cvtailro_dev.db")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:  # noqa: N802
        return self._database_uri()

    @property
    def SQLALCHEMY_ENGINE_OPTIONS(self) -> dict:  # noqa: N802
        if self._database_uri().startswith("sqlite"):
            return {}
        return {
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
        }


class DevelopmentSettings(BaseSettings):
    DEBUG = True
    SESSION_COOKIE_SECURE = False


class ProductionSettings(BaseSettings):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


class TestingSettings(BaseSettings):
    TESTING = True
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"  # type: ignore[assignment]
    SQLALCHEMY_ENGINE_OPTIONS = {}  # type: ignore[assignment]
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False  # Disable for predictable integration tests


settings_map = {
    "development": DevelopmentSettings,
    "production": ProductionSettings,
    "testing": TestingSettings,
}
