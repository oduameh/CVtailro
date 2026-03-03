"""Analytics models — durable event log and daily aggregate tables."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


def hash_user_id(user_id: str | None) -> str | None:
    """One-way hash a user ID for privacy-safe analytics storage."""
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


class AnalyticsEvent(db.Model):
    """Raw analytics event log — retained for 90 days then pruned."""

    __tablename__ = "analytics_events"
    __table_args__ = (
        db.Index("idx_ae_name_time", "event_name", "event_time"),
        db.Index("idx_ae_category_time", "category", "event_time"),
        db.Index("idx_ae_time", "event_time"),
    )

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    event_name = db.Column(db.String(100), nullable=False, index=True)
    category = db.Column(db.String(30), nullable=False, index=True)
    event_time = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    user_id_hash = db.Column(db.String(16), nullable=True)
    request_id = db.Column(db.String(20), nullable=True)
    job_id = db.Column(db.String(32), nullable=True, index=True)
    metadata_json = db.Column(db.JSON, nullable=True)


class DailyMetric(db.Model):
    """Pre-aggregated daily metrics for fast dashboard queries."""

    __tablename__ = "daily_metrics"
    __table_args__ = (
        db.UniqueConstraint("date", "metric_name", "dimension_key", name="uq_daily_metric"),
        db.Index("idx_dm_date_name", "date", "metric_name"),
    )

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    date = db.Column(db.Date, nullable=False, index=True)
    metric_name = db.Column(db.String(100), nullable=False, index=True)
    metric_value = db.Column(db.Float, nullable=False, default=0.0)
    dimension_key = db.Column(db.String(200), nullable=False, default="")
    dimensions = db.Column(db.JSON, nullable=True)
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
