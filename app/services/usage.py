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
            return len(self._prune(self._daily, key, 86400))

    def check_and_record_daily(self, key: str, limit: int, needed: int = 1) -> bool:
        """Atomically check the rolling-24h count against *limit* and record *needed*
        if allowed (anonymous/IP path). Returns False if it would exceed the limit.

        The in-memory branch is fully atomic under the lock (no check→record gap).
        The Redis branch is check-then-incr — a soft over-count is acceptable for a
        cost-control cap that is already bounded by the hourly limiter.
        """
        if limit <= 0:
            return True
        if redis_available():
            if rate_limit_count("daily_usage", key, 86400) + needed > limit:
                return False
            for _ in range(needed):
                rate_limit_incr("daily_usage", key, 86400)
            return True
        with self._lock:
            now = time.time()
            times = self._prune(self._daily, key, 86400)
            if len(times) + needed > limit:
                return False
            times.extend([now] * needed)
            self._daily[key] = times
            return True

    def _prune(self, store: dict[str, list[float]], key: str, window: int) -> list[float]:
        """Drop timestamps outside *window*; remove the key entirely if none remain
        (prevents unbounded growth of per-IP keys). Caller must hold self._lock."""
        now = time.time()
        times = [t for t in store.get(key, []) if t > now - window]
        if times:
            store[key] = times
        else:
            store.pop(key, None)
        return times

    def check_and_record(self, key: str, limit: int) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if limit <= 0:
            with self._lock:
                self._total += 1
            return True
        if redis_available():
            return rate_limit_check_incr("usage", key, limit, 3600)
        with self._lock:
            times = self._prune(self._requests, key, 3600)
            if len(times) >= limit:
                return False
            times.append(time.time())
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

    def _prune(self, ip: str) -> list[float]:
        """Drop attempts outside the window; remove the key if none remain. Holds lock."""
        now = time.time()
        attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
        if attempts:
            self._failures[ip] = attempts
        else:
            self._failures.pop(ip, None)
        return attempts

    def is_blocked(self, ip: str) -> bool:
        if redis_available():
            return rate_limit_count("login_fail", ip, self.WINDOW) >= self.MAX_ATTEMPTS
        with self._lock:
            return len(self._prune(ip)) >= self.MAX_ATTEMPTS

    def record_failure(self, ip: str) -> None:
        if redis_available():
            rate_limit_incr("login_fail", ip, self.WINDOW)
            return
        with self._lock:
            attempts = self._prune(ip)
            attempts.append(time.time())
            self._failures[ip] = attempts

    def reset(self, ip: str) -> None:
        if redis_available():
            rate_limit_reset("login_fail", ip)
            return
        with self._lock:
            self._failures.pop(ip, None)


usage_tracker = UsageTracker()
login_rate_limiter = LoginRateLimiter()
