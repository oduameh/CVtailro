"""Admin configuration management — DB-backed with in-memory cache, file fallback.

Tries the database first (AdminSetting table). Falls back to admin_config.json
for backward compatibility. Env vars always override as bootstrap defaults.
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

logger = logging.getLogger(__name__)

# Use the output directory (writable in Docker) for file fallback
_app_root = Path(__file__).resolve().parent.parent.parent
_output_dir = _app_root / "output"
CONFIG_FILE = _output_dir / "admin_config.json" if _output_dir.is_dir() else _app_root / "admin_config.json"

_SETTING_KEYS = [
    "api_key",
    "default_model",
    "allow_user_model_selection",
    "rate_limit_per_hour",
    "admin_password_hash",
    "updated_at",
    "nim_api_key",
    "active_provider",
]


@dataclass
class AdminConfig:
    api_key: str = ""
    default_model: str = ""
    allow_user_model_selection: bool = True
    rate_limit_per_hour: int = 0
    admin_password_hash: str = ""
    updated_at: str = ""
    nim_api_key: str = ""
    active_provider: str = "openrouter"  # "openrouter" or "nim"


def _try_load_from_db() -> AdminConfig | None:
    """Attempt to load settings from the admin_settings table."""
    try:
        from flask import current_app  # noqa: F401

        from app.models.admin_config import AdminSetting

        rows = AdminSetting.query.filter(AdminSetting.key.in_(_SETTING_KEYS)).all()
        if not rows:
            return None
        data: dict = {}
        for row in rows:
            data[row.key] = row.value
        config = AdminConfig()
        if "api_key" in data and data["api_key"]:
            # Reject masked/corrupted keys that were accidentally saved
            if "..." not in data["api_key"]:
                config.api_key = data["api_key"]
        if "default_model" in data and data["default_model"]:
            config.default_model = data["default_model"]
        if "allow_user_model_selection" in data:
            config.allow_user_model_selection = data["allow_user_model_selection"] != "false"
        if "rate_limit_per_hour" in data:
            try:
                config.rate_limit_per_hour = int(data["rate_limit_per_hour"])
            except (ValueError, TypeError):
                pass
        if "admin_password_hash" in data and data["admin_password_hash"]:
            config.admin_password_hash = data["admin_password_hash"]
        if "updated_at" in data:
            config.updated_at = data["updated_at"] or ""
        if "nim_api_key" in data and data["nim_api_key"] and "..." not in data["nim_api_key"]:
            config.nim_api_key = data["nim_api_key"]
        if "active_provider" in data and data["active_provider"] in ("openrouter", "nim"):
            config.active_provider = data["active_provider"]
        return config
    except Exception:
        return None


def _save_to_db(config: AdminConfig) -> bool:
    """Persist settings to the admin_settings table. Returns True on success."""
    try:
        from app.extensions import db
        from app.models.admin_config import AdminSetting

        data = {
            "api_key": config.api_key,
            "default_model": config.default_model,
            "allow_user_model_selection": str(config.allow_user_model_selection).lower(),
            "rate_limit_per_hour": str(config.rate_limit_per_hour),
            "admin_password_hash": config.admin_password_hash,
            "updated_at": config.updated_at,
            "nim_api_key": config.nim_api_key,
            "active_provider": config.active_provider,
        }
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        for key, value in data.items():
            row = AdminSetting.query.filter_by(key=key).first()
            if row:
                row.value = value
                row.updated_at = now
            else:
                row = AdminSetting(key=key, value=value, updated_at=now)
                db.session.add(row)
        db.session.commit()
        return True
    except Exception as e:
        logger.debug(f"DB save failed (falling back to file): {e}")
        try:
            from app.extensions import db

            db.session.rollback()
        except Exception:
            pass
        return False


class AdminConfigManager:
    _lock = threading.Lock()
    _cache: AdminConfig | None = None
    _cache_time: float = 0
    CACHE_TTL = 5

    @classmethod
    def load(cls) -> AdminConfig:
        with cls._lock:
            now = time.time()
            if cls._cache and (now - cls._cache_time) < cls.CACHE_TTL:
                return cls._cache

            # Try DB first, then fall back to file
            config = _try_load_from_db()

            if config is None:
                config = AdminConfig()
                if CONFIG_FILE.exists():
                    try:
                        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                        config = AdminConfig(**{k: v for k, v in data.items() if hasattr(AdminConfig, k)})
                    except Exception as e:
                        logger.error(f"Failed to load admin config from file: {e}")

            # Env var fallbacks
            env_key = os.environ.get("OPENROUTER_API_KEY", "")
            if env_key and not config.api_key:
                config.api_key = env_key
            env_nim = os.environ.get("NVIDIA_NIM_API_KEY", "")
            if env_nim and not config.nim_api_key:
                config.nim_api_key = env_nim
            env_pw = os.environ.get("ADMIN_PASSWORD", "")
            if env_pw and not config.admin_password_hash:
                config.admin_password_hash = cls._hash_password(env_pw)

            cls._cache = config
            cls._cache_time = now
            return config

    @classmethod
    def save(cls, config: AdminConfig) -> None:
        with cls._lock:
            config.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Try DB first
            if not _save_to_db(config):
                # Fall back to file
                try:
                    CONFIG_FILE.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
                except OSError as e:
                    logger.error(f"Failed to write admin config to disk: {e}")
                    raise

            cls._cache = config
            cls._cache_time = time.time()

    @classmethod
    def is_configured(cls) -> bool:
        return bool(cls.load().api_key.strip())

    @classmethod
    def verify_password(cls, password: str) -> bool:
        env_pw = os.environ.get("ADMIN_PASSWORD", "")
        if env_pw:
            import hmac

            if hmac.compare_digest(password, env_pw):
                return True
        config = cls.load()
        if not config.admin_password_hash:
            return False
        if config.admin_password_hash.startswith("pbkdf2:"):
            return check_password_hash(config.admin_password_hash, password)
        # Legacy SHA-256 — verify and auto-upgrade
        if hashlib.sha256(password.encode()).hexdigest() == config.admin_password_hash:
            config.admin_password_hash = cls._hash_password(password)
            try:
                cls.save(config)
                logger.info("Auto-upgraded admin password hash from SHA-256 to pbkdf2")
            except Exception:
                pass
            return True
        return False

    @classmethod
    def has_password(cls) -> bool:
        if os.environ.get("ADMIN_PASSWORD", ""):
            return True
        return bool(cls.load().admin_password_hash)

    @staticmethod
    def _hash_password(password: str) -> str:
        return generate_password_hash(password, method="pbkdf2:sha256:600000")
