"""Unit tests for usage tracking and login rate limiting."""

import pytest

from app.services.usage import LoginRateLimiter, UsageTracker


@pytest.mark.unit
class TestUsageTracker:
    """UsageTracker — hourly rate limiting per key."""

    def test_check_and_record_allows_when_under_limit(self):
        tracker = UsageTracker()
        assert tracker.check_and_record("user:1", 10) is True
        assert tracker.check_and_record("user:1", 10) is True

    def test_check_and_record_blocks_when_over_limit(self):
        tracker = UsageTracker()
        limit = 3
        for _ in range(limit):
            assert tracker.check_and_record("user:1", limit) is True
        assert tracker.check_and_record("user:1", limit) is False

    def test_check_and_record_per_key_isolation(self):
        tracker = UsageTracker()
        limit = 2
        for _ in range(limit):
            assert tracker.check_and_record("user:A", limit) is True
        assert tracker.check_and_record("user:A", limit) is False
        assert tracker.check_and_record("user:B", limit) is True

    def test_check_and_record_zero_limit_allows_all(self):
        tracker = UsageTracker()
        assert tracker.check_and_record("user:1", 0) is True
        assert tracker.check_and_record("user:1", 0) is True

    def test_get_stats_returns_dict(self):
        tracker = UsageTracker()
        tracker.check_and_record("user:1", 5)
        stats = tracker.get_stats()
        assert "total_requests" in stats
        assert "requests_last_hour" in stats
        assert "active_sessions" in stats
        assert stats["total_requests"] >= 1


@pytest.mark.unit
class TestLoginRateLimiter:
    """LoginRateLimiter — brute-force protection."""

    def test_initially_not_blocked(self):
        limiter = LoginRateLimiter()
        assert limiter.is_blocked("192.168.1.1") is False

    def test_blocks_after_max_attempts(self):
        limiter = LoginRateLimiter()
        for _ in range(LoginRateLimiter.MAX_ATTEMPTS):
            limiter.record_failure("10.0.0.1")
        assert limiter.is_blocked("10.0.0.1") is True

    def test_reset_clears_block(self):
        limiter = LoginRateLimiter()
        for _ in range(LoginRateLimiter.MAX_ATTEMPTS):
            limiter.record_failure("10.0.0.2")
        assert limiter.is_blocked("10.0.0.2") is True
        limiter.reset("10.0.0.2")
        assert limiter.is_blocked("10.0.0.2") is False

    def test_per_ip_isolation(self):
        limiter = LoginRateLimiter()
        for _ in range(LoginRateLimiter.MAX_ATTEMPTS):
            limiter.record_failure("10.0.0.3")
        assert limiter.is_blocked("10.0.0.3") is True
        assert limiter.is_blocked("10.0.0.4") is False
