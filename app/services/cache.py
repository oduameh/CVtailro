"""Optional Redis integration — graceful fallback to no-op when unavailable.

Provides a thin wrapper for caching, session storage, and pub/sub.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = False


def init_redis() -> bool:
    """Try to connect to Redis. Returns True if successful."""
    global _redis_client, _redis_available

    redis_url = os.environ.get("REDIS_URL", "")
    if not redis_url:
        logger.info("REDIS_URL not set — running without Redis")
        return False

    try:
        import redis

        _redis_client = redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connected successfully")
        return True
    except ImportError:
        logger.info("redis package not installed — running without Redis")
        return False
    except Exception as e:
        logger.warning(f"Redis connection failed — running without Redis: {e}")
        _redis_client = None
        _redis_available = False
        return False


def get_redis():
    """Return the Redis client, or None if unavailable."""
    return _redis_client if _redis_available else None


def is_available() -> bool:
    return _redis_available


def cache_get(key: str) -> str | None:
    if not _redis_available or _redis_client is None:
        return None
    try:
        return _redis_client.get(key)
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    if not _redis_available or _redis_client is None:
        return False
    try:
        _redis_client.setex(key, ttl, str(value))
        return True
    except Exception:
        return False


def cache_delete(key: str) -> bool:
    if not _redis_available or _redis_client is None:
        return False
    try:
        _redis_client.delete(key)
        return True
    except Exception:
        return False
