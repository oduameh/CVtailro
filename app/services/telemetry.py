"""Telemetry service — unified event emission with PII redaction.

All analytics events flow through ``track()`` which:
1. Redacts PII (emails, resume text, secrets)
2. Persists to the analytics_events table (fire-and-forget, never blocks callers)
3. Optionally forwards to external sinks (Sentry breadcrumbs, structured log)

Usage:
    from app.services.telemetry import track
    track("tailor.job.completed", category="tailor", job_id=jid, metadata={...})
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any

from flask import g, has_request_context

logger = logging.getLogger("cvtailro.telemetry")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_REDACT_KEYS = frozenset({
    "email", "password", "api_key", "secret", "token",
    "resume_text", "original_resume_text", "job_description_full",
    "job_description", "resume_md", "ats_resume_md", "cover_letter_md",
    "talking_points_md", "password_hash",
})


def _redact_value(key: str, value: Any) -> Any:
    """Strip PII from metadata values before storage."""
    if key.lower() in _REDACT_KEYS:
        if isinstance(value, str):
            return f"[REDACTED:{len(value)}chars]"
        return "[REDACTED]"
    if isinstance(value, str):
        value = _EMAIL_RE.sub("[EMAIL_REDACTED]", value)
    return value


def _sanitize_metadata(data: dict | None) -> dict | None:
    if not data:
        return data
    return {k: _redact_value(k, v) for k, v in data.items()}


def _get_request_id() -> str | None:
    if has_request_context():
        return getattr(g, "request_id", None)
    return None


def track(
    event_name: str,
    *,
    category: str = "system",
    user_id: str | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Record an analytics event (non-blocking).

    Safe to call from any context (request handler, background thread, etc.).
    DB writes happen in a short-lived thread so callers are never blocked.
    """
    rid = request_id or _get_request_id()
    safe_meta = _sanitize_metadata(metadata)

    # Structured log line (always emitted, works even if DB is down)
    logger.info(
        "event=%s category=%s job_id=%s rid=%s meta=%s",
        event_name, category, job_id or "-", rid or "-",
        {k: v for k, v in (safe_meta or {}).items() if k not in ("traceback",)} if safe_meta else "-",
    )

    # Persist to DB in background thread (fire-and-forget)
    threading.Thread(
        target=_persist_event,
        args=(event_name, category, user_id, job_id, rid, safe_meta),
        daemon=True,
    ).start()


def _persist_event(
    event_name: str,
    category: str,
    user_id: str | None,
    job_id: str | None,
    request_id: str | None,
    metadata: dict | None,
) -> None:
    """Write event row to analytics_events (runs in daemon thread)."""
    try:
        # We need an app context for DB access
        from flask import current_app

        from app.extensions import db
        from app.models.analytics import AnalyticsEvent, hash_user_id
        try:
            app = current_app._get_current_object()
        except RuntimeError:
            # No app context available (e.g. during startup). Try to get it
            # from the global app reference.
            return

        with app.app_context():
            evt = AnalyticsEvent(
                event_name=event_name,
                category=category,
                event_time=datetime.now(timezone.utc),
                user_id_hash=hash_user_id(user_id),
                request_id=request_id,
                job_id=job_id,
                metadata_json=metadata,
            )
            db.session.add(evt)
            db.session.commit()
    except Exception:
        logger.debug("Failed to persist analytics event %s", event_name, exc_info=True)


def track_with_app(
    app,
    event_name: str,
    *,
    category: str = "system",
    user_id: str | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Like track() but accepts an explicit Flask app (for background threads)."""
    rid = request_id or _get_request_id()
    safe_meta = _sanitize_metadata(metadata)

    logger.info(
        "event=%s category=%s job_id=%s rid=%s",
        event_name, category, job_id or "-", rid or "-",
    )

    threading.Thread(
        target=_persist_event_with_app,
        args=(app, event_name, category, user_id, job_id, rid, safe_meta),
        daemon=True,
    ).start()


def _persist_event_with_app(
    app,
    event_name: str,
    category: str,
    user_id: str | None,
    job_id: str | None,
    request_id: str | None,
    metadata: dict | None,
) -> None:
    """Write event row using an explicit app context."""
    try:
        from app.extensions import db
        from app.models.analytics import AnalyticsEvent, hash_user_id

        with app.app_context():
            evt = AnalyticsEvent(
                event_name=event_name,
                category=category,
                event_time=datetime.now(timezone.utc),
                user_id_hash=hash_user_id(user_id),
                request_id=request_id,
                job_id=job_id,
                metadata_json=metadata,
            )
            db.session.add(evt)
            db.session.commit()
    except Exception:
        logger.debug("Failed to persist analytics event %s", event_name, exc_info=True)
