"""TailoringJob and JobFile models — pipeline results and R2 file metadata."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class TailoringJob(db.Model):
    __tablename__ = "tailoring_jobs"
    __table_args__ = (
        db.Index("idx_tailoring_jobs_user_created", "user_id", "created_at"),
        db.Index("idx_tailoring_jobs_user_status", "user_id", "status"),
    )

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(32), db.ForeignKey("users.id"), nullable=True, index=True)
    status = db.Column(db.String(20), nullable=False, default="running")
    job_title = db.Column(db.String(500), nullable=True)
    company = db.Column(db.String(500), nullable=True)
    match_score = db.Column(db.Float, nullable=True)
    original_match_score = db.Column(db.Float, nullable=True)
    cosine_similarity = db.Column(db.Float, nullable=True)
    missing_keywords = db.Column(db.JSON, nullable=True)
    rewrite_mode = db.Column(db.String(20), nullable=False, default="conservative")
    template = db.Column(db.String(20), nullable=False, default="modern")
    model_used = db.Column(db.String(255), nullable=True)
    job_description_snippet = db.Column(db.String(500), nullable=True)
    job_description_full = db.Column(db.Text, nullable=True)
    original_resume_text = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    ats_resume_md = db.Column(db.Text, nullable=True)
    recruiter_resume_md = db.Column(db.Text, nullable=True)
    talking_points_md = db.Column(db.Text, nullable=True)
    cover_letter_md = db.Column(db.Text, nullable=True)
    section_scores = db.Column(db.JSON, nullable=True)
    resume_quality_json = db.Column(db.JSON, nullable=True)
    ats_check_json = db.Column(db.JSON, nullable=True)
    email_templates_md = db.Column(db.Text, nullable=True)
    keyword_density_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)

    user = db.relationship("User", back_populates="jobs")
    files = db.relationship(
        "JobFile",
        back_populates="job",
        lazy="selectin",
        cascade="all, delete-orphan",
    )


class JobFile(db.Model):
    __tablename__ = "job_files"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    job_id = db.Column(
        db.String(32),
        db.ForeignKey("tailoring_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    r2_key = db.Column(db.String(1024), nullable=False)
    content_type = db.Column(db.String(100), nullable=False, default="application/octet-stream")
    size_bytes = db.Column(db.Integer, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    job = db.relationship("TailoringJob", back_populates="files")


class JobApplication(db.Model):
    """Track job applications with status and linked tailoring jobs."""

    __tablename__ = "job_applications"
    __table_args__ = (db.Index("idx_job_applications_user", "user_id"),)

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(32), db.ForeignKey("users.id"), nullable=False, index=True)
    tailoring_job_id = db.Column(db.String(32), db.ForeignKey("tailoring_jobs.id"), nullable=True)
    company = db.Column(db.String(500), nullable=False)
    job_title = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), nullable=False, default="saved")
    url = db.Column(db.String(2000), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    applied_date = db.Column(db.DateTime(timezone=True), nullable=True)
    interview_date = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship("User", backref="job_applications")
    tailoring_job = db.relationship("TailoringJob", backref="application")
