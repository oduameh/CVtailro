"""Unit tests for file service helpers (can_access_job logic)."""

import pytest

from app.services.file_service import _can_access_job


@pytest.mark.unit
class TestCanAccessJob:
    """_can_access_job — job ownership verification."""

    def test_anonymous_job_accessible_by_anyone(self):
        """Job with no owner (user_id=None) is accessible by anyone."""
        assert _can_access_job(None, None, False) is True
        assert _can_access_job(None, "user-123", True) is True

    def test_user_job_denied_to_anonymous(self):
        """User-owned job is denied to anonymous."""
        assert _can_access_job("user-123", None, False) is False

    def test_user_job_allowed_to_owner(self):
        """User-owned job is allowed to owner."""
        assert _can_access_job("user-123", "user-123", True) is True

    def test_user_job_denied_to_other_user(self):
        """User-owned job is denied to different user."""
        assert _can_access_job("user-123", "user-456", True) is False
