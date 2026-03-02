"""Backward-compatible re-export — new code should import from app.routes.saved_resumes."""

from app.routes.saved_resumes import saved_resumes_bp

__all__ = ["saved_resumes_bp"]
