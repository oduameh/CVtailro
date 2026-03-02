"""Backward-compatible re-exports — new code should import from app.models and app.extensions."""

from app.extensions import db
from app.models.user import User
from app.models.job import TailoringJob, JobFile
from app.models.saved_resume import SavedResume

__all__ = ["db", "User", "TailoringJob", "JobFile", "SavedResume"]
