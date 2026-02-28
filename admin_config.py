"""Admin configuration management for CVtailro."""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / "admin_config.json"


@dataclass
class AdminConfig:
    api_key: str = ""
    default_model: str = ""
    allow_user_model_selection: bool = True
    rate_limit_per_hour: int = 0  # 0 = unlimited
    admin_password_hash: str = ""
    updated_at: str = ""


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
            config = AdminConfig()
            if CONFIG_FILE.exists():
                try:
                    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                    config = AdminConfig(**{k: v for k, v in data.items() if hasattr(AdminConfig, k)})
                except Exception as e:
                    logger.error(f"Failed to load admin config: {e}")
            # Env var fallbacks
            env_key = os.environ.get("OPENROUTER_API_KEY", "")
            if env_key and not config.api_key:
                config.api_key = env_key
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
            CONFIG_FILE.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
            cls._cache = config
            cls._cache_time = time.time()

    @classmethod
    def is_configured(cls) -> bool:
        return bool(cls.load().api_key.strip())

    @classmethod
    def verify_password(cls, password: str) -> bool:
        # Check against ADMIN_PASSWORD env var directly
        env_pw = os.environ.get("ADMIN_PASSWORD", "")
        if env_pw and password == env_pw:
            return True
        # Check against stored hash
        config = cls.load()
        if not config.admin_password_hash:
            return False
        return cls._hash_password(password) == config.admin_password_hash

    @classmethod
    def has_password(cls) -> bool:
        if os.environ.get("ADMIN_PASSWORD", ""):
            return True
        return bool(cls.load().admin_password_hash)

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
