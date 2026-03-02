"""SavedResume model — user's reusable master resumes."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class SavedResume(db.Model):
    __tablename__ = "saved_resumes"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    user_id = db.Column(
        db.String(32), db.ForeignKey("users.id"), nullable=False, index=True
    )
    name = db.Column(db.String(255), nullable=False, default="My Resume")
    resume_text = db.Column(db.Text, nullable=False)
    file_hash = db.Column(db.String(64), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = db.relationship(
        "User", backref=db.backref("saved_resumes", lazy="dynamic")
    )
