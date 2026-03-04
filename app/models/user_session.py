"""UserSession model — server-side session tracking for authenticated users."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class UserSession(db.Model):
    __tablename__ = "user_sessions"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(32), db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)
    device_type = db.Column(db.String(20), nullable=True)
    browser_name = db.Column(db.String(50), nullable=True)
    os_name = db.Column(db.String(50), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_activity_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    revoked_at = db.Column(db.DateTime(timezone=True), nullable=True)
    revoked_reason = db.Column(db.String(50), nullable=True)

    user = db.relationship("User", backref=db.backref("sessions", lazy="dynamic"))

    def __repr__(self) -> str:
        return f"<UserSession {self.id[:8]} user={self.user_id[:8]} active={self.is_active}>"
