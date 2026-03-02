"""Usage tracking and login rate limiting — thread-safe singletons."""

import threading
import time


class UsageTracker:
    """Track API usage per key (user or IP) with hourly rate limiting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = {}
        self._total = 0

    def check_and_record(self, key: str, limit: int) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if limit <= 0:
            with self._lock:
                self._total += 1
            return True
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
        with self._lock:
            now = time.time()
            hour_ago = now - 3600
            active = sum(1 for ts in self._requests.values() if any(t > hour_ago for t in ts))
            recent = sum(len([t for t in ts if t > hour_ago]) for ts in self._requests.values())
            return {
                "total_requests": self._total,
                "requests_last_hour": recent,
                "active_sessions": active,
            }


class LoginRateLimiter:
    """Brute-force protection for admin login — block after MAX_ATTEMPTS failures."""

    MAX_ATTEMPTS = 5
    WINDOW = 900  # 15 minutes

    def __init__(self):
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    def is_blocked(self, ip: str) -> bool:
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            self._failures[ip] = attempts
            return len(attempts) >= self.MAX_ATTEMPTS

    def record_failure(self, ip: str) -> None:
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            attempts.append(now)
            self._failures[ip] = attempts

    def reset(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)


usage_tracker = UsageTracker()
login_rate_limiter = LoginRateLimiter()
