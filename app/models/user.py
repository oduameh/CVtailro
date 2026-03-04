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

    # Session invalidation: bumped on password change to revoke all other sessions
    session_version = db.Column(db.Integer, nullable=False, default=0)

    # Per-user account lockout (brute-force protection beyond IP-based limiting)
    failed_login_attempts = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime(timezone=True), nullable=True)

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

    @property
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    def record_failed_login(self) -> None:
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        if self.failed_login_attempts >= 10:
            from datetime import timedelta
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)

    def reset_failed_logins(self) -> None:
        self.failed_login_attempts = 0
        self.locked_until = None
