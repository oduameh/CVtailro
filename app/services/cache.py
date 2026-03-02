"""Optional Redis integration — graceful fallback to no-op when unavailable.

Provides a thin wrapper for caching, session storage, and pub/sub.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = False


def init_redis() -> bool:
    """Try to connect to Redis. Returns True if successful."""
    global _redis_client, _redis_available

    redis_url = (os.environ.get("REDIS_URL") or "").strip()
    if not redis_url or redis_url.startswith("memory://"):
        logger.info("REDIS_URL not set or memory:// — running without Redis")
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


# ── Redis-backed rate limiting (sliding window) ─────────────────────────────


def rate_limit_check_incr(prefix: str, key: str, limit: int, window_seconds: int) -> bool:
    """
    Sliding-window rate limit: allow if under limit, else deny.
    Records the request when allowed. Returns True if allowed, False if blocked.
    """
    if not _redis_available or _redis_client is None:
        return True
    try:
        now = int(1000 * time.time())
        window_start = now - (window_seconds * 1000)
        rkey = f"{prefix}:{key}"
        pipe = _redis_client.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", window_start)
        pipe.zcard(rkey)
        result = pipe.execute()
        count = result[1]
        if count >= limit:
            return False
        pipe = _redis_client.pipeline()
        pipe.zadd(rkey, {str(uuid.uuid4()): now})
        pipe.expire(rkey, window_seconds + 60)
        pipe.execute()
        return True
    except Exception:
        return True


def rate_limit_count(prefix: str, key: str, window_seconds: int) -> int:
    """Return count of entries in the sliding window."""
    if not _redis_available or _redis_client is None:
        return 0
    try:
        now = int(1000 * time.time())
        window_start = now - (window_seconds * 1000)
        rkey = f"{prefix}:{key}"
        _redis_client.zremrangebyscore(rkey, "-inf", window_start)
        return _redis_client.zcard(rkey)
    except Exception:
        return 0


def rate_limit_incr(prefix: str, key: str, window_seconds: int) -> int:
    """Add one entry and return the new count."""
    if not _redis_available or _redis_client is None:
        return 0
    try:
        now = int(1000 * time.time())
        window_start = now - (window_seconds * 1000)
        rkey = f"{prefix}:{key}"
        pipe = _redis_client.pipeline()
        pipe.zremrangebyscore(rkey, "-inf", window_start)
        pipe.zadd(rkey, {str(uuid.uuid4()): now})
        pipe.expire(rkey, window_seconds + 60)
        pipe.zcard(rkey)
        result = pipe.execute()
        return result[-1]
    except Exception:
        return 0


def rate_limit_reset(prefix: str, key: str) -> None:
    """Clear all entries for the given key."""
    if not _redis_available or _redis_client is None:
        return
    try:
        _redis_client.delete(f"{prefix}:{key}")
    except Exception:
        pass
