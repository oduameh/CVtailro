"""User model — Google OAuth + email/password users with admin flag."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask_login import UserMixin

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    google_id = db.Column(db.String(255), unique=True, nullable=True, index=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False, default="")
    picture_url = db.Column(db.String(1024), nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

    # Email/password auth fields
    password_hash = db.Column(db.String(255), nullable=True)
    email_verified = db.Column(db.Boolean, nullable=False, default=False)
    email_verified_at = db.Column(db.DateTime(timezone=True), nullable=True)
    auth_provider = db.Column(db.String(20), nullable=False, default="google")

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_login_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    jobs = db.relationship(
        "TailoringJob",
        back_populates="user",
        lazy="dynamic",
        order_by="TailoringJob.created_at.desc()",
    )

    @property
    def has_password(self) -> bool:
        """Return True if the user has a password set."""
        return self.password_hash is not None
