"""AdminSetting model — key-value store for admin configuration in the database.

Replaces the file-based admin_config.json for multi-instance deployments.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


class AdminSetting(db.Model):
    """Persistent admin settings stored in the database."""

    __tablename__ = "admin_settings"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    key = db.Column(db.String(255), unique=True, nullable=False, index=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
