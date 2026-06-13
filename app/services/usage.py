"""Usage tracking and login rate limiting — Redis-backed when available."""

from __future__ import annotations

import threading
import time

from app.services.cache import (
    is_available as redis_available,
)
from app.services.cache import (
    rate_limit_check_incr,
    rate_limit_count,
    rate_limit_incr,
    rate_limit_reset,
)


def daily_jobs_used(user_id: str) -> int:
    """Count an authenticated user's tailoring jobs since UTC midnight.

    Counts all statuses — an errored job still consumed API spend. Durable
    across restarts (reads the TailoringJob table, served by the
    idx_tailoring_jobs_user_created index), so no Redis dependency.
    """
    from datetime import datetime, timezone

    from app.models import TailoringJob

    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return TailoringJob.query.filter(
        TailoringJob.user_id == user_id,
        TailoringJob.created_at >= midnight,
    ).count()


class UsageTracker:
    """Track API usage per key (user or IP) with hourly rate limiting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = {}
        self._daily: dict[str, list[float]] = {}
        self._total = 0

    def daily_count(self, key: str) -> int:
        """Jobs recorded for *key* in the rolling 24h window (anonymous/IP path)."""
        if redis_available():
            return rate_limit_count("daily_usage", key, 86400)
        with self._lock:
            now = time.time()
            times = [t for t in self._daily.get(key, []) if t > now - 86400]
            self._daily[key] = times
            return len(times)

    def record_daily(self, key: str) -> None:
        """Record one job for *key* in the rolling 24h window (anonymous/IP path)."""
        if redis_available():
            rate_limit_incr("daily_usage", key, 86400)
            return
        with self._lock:
            now = time.time()
            times = [t for t in self._daily.get(key, []) if t > now - 86400]
            times.append(now)
            self._daily[key] = times

    def check_and_record(self, key: str, limit: int) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if limit <= 0:
            with self._lock:
                self._total += 1
            return True
        if redis_available():
            return rate_limit_check_incr("usage", key, limit, 3600)
        with self._lock:
            now = time.time()
            times = [t for t in self._requests.get(key, []) if t > now - 3600]
            if len(times) >= limit:
                return False
            times.append(now)
            self._requests[key] = times
            self._total += 1
            return True

    def get_stats(self) -> dict:
        if redis_available():
            return {
                "total_requests": 0,
                "requests_last_hour": 0,
                "active_sessions": 0,
                "backend": "redis",
            }
        with self._lock:
            now = time.time()
            hour_ago = now - 3600
            active = sum(1 for ts in self._requests.values() if any(t > hour_ago for t in ts))
            recent = sum(len([t for t in ts if t > hour_ago]) for ts in self._requests.values())
            return {
                "total_requests": self._total,
                "requests_last_hour": recent,
                "active_sessions": active,
                "backend": "memory",
            }


class LoginRateLimiter:
    """Brute-force protection for admin login — block after MAX_ATTEMPTS failures."""

    MAX_ATTEMPTS = 5
    WINDOW = 900  # 15 minutes

    def __init__(self):
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def is_blocked(self, ip: str) -> bool:
        if redis_available():
            return rate_limit_count("login_fail", ip, self.WINDOW) >= self.MAX_ATTEMPTS
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            self._failures[ip] = attempts
            return len(attempts) >= self.MAX_ATTEMPTS

    def record_failure(self, ip: str) -> None:
        if redis_available():
            rate_limit_incr("login_fail", ip, self.WINDOW)
            return
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            attempts.append(now)
            self._failures[ip] = attempts

    def reset(self, ip: str) -> None:
        if redis_available():
            rate_limit_reset("login_fail", ip)
            return
        with self._lock:
            self._failures.pop(ip, None)


usage_tracker = UsageTracker()
login_rate_limiter = LoginRateLimiter()
