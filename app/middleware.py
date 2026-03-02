"""Request middleware — structured logging, request IDs, and Sentry integration."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

from flask import Flask, g, request


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def init_structured_logging(flask_app: Flask) -> None:
    """Configure structured JSON logging for production, plain text for dev."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    env = os.environ.get("FLASK_ENV", "production")
    if env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    handler.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(handler)


def init_request_id(flask_app: Flask) -> None:
    """Attach a unique request ID to every request for tracing."""

    @flask_app.before_request
    def _set_request_id():
        g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        g.request_start = time.time()

    @flask_app.after_request
    def _log_request(response):
        if request.path == "/api/health":
            return response
        duration_ms = (time.time() - getattr(g, "request_start", time.time())) * 1000
        logger = logging.getLogger("cvtailro.access")
        logger.info(
            f"{request.method} {request.path} {response.status_code} "
            f"{duration_ms:.0f}ms rid={getattr(g, 'request_id', '-')}"
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response


def init_sentry(flask_app: Flask) -> None:
    """Initialize Sentry error tracking if SENTRY_DSN is set."""
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration

        sentry_sdk.init(
            dsn=dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.1,
            environment=os.environ.get("FLASK_ENV", "production"),
            send_default_pii=False,
        )
        logging.getLogger("cvtailro").info("Sentry error tracking enabled")
    except ImportError:
        logging.getLogger("cvtailro").warning(
            "sentry-sdk not installed — Sentry disabled"
        )
    except Exception as e:
        logging.getLogger("cvtailro").warning(f"Sentry init failed: {e}")
