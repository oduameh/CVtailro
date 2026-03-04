"""LoginEvent model — immutable log of all authentication events."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class LoginEvent(db.Model):
    __tablename__ = "login_events"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    user_id = db.Column(db.String(32), db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = db.Column(db.String(255), nullable=True)
    event_type = db.Column(db.String(30), nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)
    device_type = db.Column(db.String(20), nullable=True)
    browser_name = db.Column(db.String(50), nullable=True)
    os_name = db.Column(db.String(50), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    success = db.Column(db.Boolean, nullable=False, default=True)
    failure_reason = db.Column(db.String(100), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    user = db.relationship("User", backref=db.backref("login_events", lazy="dynamic"))

    def __repr__(self) -> str:
        return f"<LoginEvent {self.event_type} user={self.user_id} at={self.created_at}>"
