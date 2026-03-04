"""Admin panel routes — configuration, analytics, user management, system stats."""

from __future__ import annotations

import logging
import os
import platform as platform_mod
import resource as resource_mod
import threading
from datetime import datetime, timedelta, timezone

import requests as http_requests
from flask import Blueprint, jsonify, render_template, request, session
from flask_login import current_user
from sqlalchemy import case, func, text

from analytics import pipeline_analytics
from app.extensions import csrf, db, limiter
from app.models import AnalyticsEvent, SavedResume, TailoringJob, User
from app.models.job import JobApplication
from app.services.admin_config import AdminConfigManager
from app.services.pipeline import (
    MAX_CONCURRENT_PIPELINES,
    MAX_QUEUE_DEPTH,
    pipeline_errors,
    pipeline_errors_lock,
    pipeline_queue_depth,
    pipeline_queue_lock,
    pipeline_semaphore,
)
from app.services.telemetry import track
from app.services.usage import login_rate_limiter, usage_tracker

logger = logging.getLogger("cvtailro.admin")

admin_bp = Blueprint("admin", __name__)
csrf.exempt(admin_bp)  # Admin uses session-based auth, all endpoints are JSON API


# ── Shared query helpers ─────────────────────────────────────────────────────


def _time_windows() -> dict[str, datetime]:
    """Return commonly used UTC time boundaries."""
    now = datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "now": now,
        "today": today,
        "week": today - timedelta(days=7),
        "month": today - timedelta(days=30),
    }


def _jobs_by_status(since: datetime | None = None) -> dict[str, int]:
    """Count jobs grouped by status, optionally filtered to a time window."""
    q = db.session.query(TailoringJob.status, func.count(TailoringJob.id))
    if since is not None:
        q = q.filter(TailoringJob.created_at >= since)
    return dict(q.group_by(TailoringJob.status).all())


def _live_pipeline_state() -> dict:
    """Return current pipeline concurrency and queue depth."""
    try:
        available = pipeline_semaphore._value
        active = MAX_CONCURRENT_PIPELINES - available
    except AttributeError:
        active = -1
    with pipeline_queue_lock:
        queue_depth = pipeline_queue_depth
    return {
        "active_pipelines": active,
        "queue_depth": queue_depth,
        "max_concurrent": MAX_CONCURRENT_PIPELINES,
        "max_queue": MAX_QUEUE_DEPTH,
    }


def _admin_required(f):
    """Decorator checking session-based admin auth OR logged-in admin user."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("admin_authenticated"):
            return f(*args, **kwargs)
        if current_user.is_authenticated and current_user.is_admin:
            return f(*args, **kwargs)
        return jsonify({"error": "Not authenticated"}), 401

    return decorated


@admin_bp.route("/admin")
def admin_page():
    return render_template("admin.html")


@admin_bp.route("/admin/api/login", methods=["POST"])
@limiter.limit("5 per minute")
def admin_login():
    client_ip = request.remote_addr or "unknown"
    if login_rate_limiter.is_blocked(client_ip):
        return jsonify({"error": "Too many failed attempts. Try again in 15 minutes."}), 429

    data = request.get_json(force=True)
    password = data.get("password", "")
    if not password:
        return jsonify({"error": "Password is required"}), 400

    if not AdminConfigManager.has_password():
        return jsonify(
            {"error": "Admin password not configured. Set the ADMIN_PASSWORD environment variable."}
        ), 503

    if AdminConfigManager.verify_password(password):
        session["admin_authenticated"] = True
        login_rate_limiter.reset(client_ip)
        track("admin.login", category="admin", metadata={"method": "password"})
        return jsonify({"ok": True})

    login_rate_limiter.record_failure(client_ip)
    track("admin.login.failed", category="admin", metadata={"method": "password"})
    return jsonify({"error": "Invalid password"}), 401


@admin_bp.route("/admin/api/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_authenticated", None)
    track("admin.logout", category="admin")
    return jsonify({"ok": True})


@admin_bp.route("/admin/api/config", methods=["GET"])
@_admin_required
def admin_get_config():
    config = AdminConfigManager.load()
    masked_key = ""
    if config.api_key:
        masked_key = config.api_key[:8] + "..." if len(config.api_key) > 8 else config.api_key
    masked_nim = ""
    if config.nim_api_key:
        masked_nim = config.nim_api_key[:10] + "..." if len(config.nim_api_key) > 10 else config.nim_api_key
    return jsonify(
        {
            "api_key": masked_key,
            "nim_api_key": masked_nim,
            "active_provider": config.active_provider,
            "default_model": config.default_model,
            "allow_user_model_selection": config.allow_user_model_selection,
            "rate_limit_per_hour": config.rate_limit_per_hour,
            "updated_at": config.updated_at,
        }
    )


@admin_bp.route("/admin/api/config", methods=["POST"])
@_admin_required
def admin_save_config():
    data = request.get_json(force=True)
    config = AdminConfigManager.load()

    if "api_key" in data:
        new_key = data["api_key"].strip()
        if new_key and "..." not in new_key:
            config.api_key = new_key
    if "nim_api_key" in data:
        new_nim = data["nim_api_key"].strip()
        if new_nim and "..." not in new_nim:
            config.nim_api_key = new_nim
    if "active_provider" in data and data["active_provider"] in ("openrouter", "nim"):
        config.active_provider = data["active_provider"]
    if "default_model" in data:
        config.default_model = data["default_model"]
    if "allow_user_model_selection" in data:
        config.allow_user_model_selection = bool(data["allow_user_model_selection"])
    if "rate_limit_per_hour" in data:
        config.rate_limit_per_hour = int(data["rate_limit_per_hour"])

    try:
        AdminConfigManager.save(config)
    except Exception as e:
        logger.exception("Failed to save admin config")
        return jsonify({"error": f"Save failed: {e}"}), 500
    changed_keys = [
        k
        for k in ("api_key", "default_model", "allow_user_model_selection", "rate_limit_per_hour")
        if k in data
    ]
    track("admin.config.updated", category="admin", metadata={"changed_keys": changed_keys})
    return jsonify({"ok": True, "updated_at": config.updated_at})


@admin_bp.route("/admin/api/test-key", methods=["POST"])
@_admin_required
def admin_test_key():
    data = request.get_json(force=True)
    api_key = data.get("api_key", "").strip()
    provider = data.get("provider", "openrouter")
    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})
    try:
        if provider == "nim":
            # Test NVIDIA NIM key
            r = http_requests.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        else:
            # Test OpenRouter key
            r = http_requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        return jsonify(
            {"valid": r.status_code == 200}
            | ({"error": f"HTTP {r.status_code}"} if r.status_code != 200 else {})
        )
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@admin_bp.route("/admin/api/errors")
@_admin_required
def admin_errors():
    with pipeline_errors_lock:
        return jsonify({"errors": list(reversed(pipeline_errors))})


@admin_bp.route("/admin/api/usage")
@_admin_required
def admin_usage():
    return jsonify(usage_tracker.get_stats())


@admin_bp.route("/admin/api/analytics")
@_admin_required
def admin_analytics():
    """Returns in-memory pipeline analytics (tokens, cost) + DB-backed stats for charts."""
    tw = _time_windows()
    pipeline_stats = pipeline_analytics.get_global_stats()

    jobs_per_day = (
        db.session.query(
            func.date(TailoringJob.created_at).label("day"),
            func.count(TailoringJob.id).label("count"),
        )
        .filter(TailoringJob.created_at >= tw["month"])
        .group_by(func.date(TailoringJob.created_at))
        .order_by(func.date(TailoringJob.created_at))
        .all()
    )

    status = _jobs_by_status()
    completed_count = status.get("complete", 0)
    avg_duration = (
        db.session.query(func.avg(TailoringJob.duration_seconds))
        .filter(TailoringJob.status == "complete", TailoringJob.duration_seconds.isnot(None))
        .scalar()
    )

    return jsonify(
        {
            **pipeline_stats,
            "jobs_over_time": [
                {"date": str(d).split()[0] if d else "1970-01-01", "jobs": c} for d, c in jobs_per_day
            ],
            "jobs_by_status": status,
            "jobs_today": TailoringJob.query.filter(TailoringJob.created_at >= tw["today"]).count(),
            "jobs_this_week": TailoringJob.query.filter(TailoringJob.created_at >= tw["week"]).count(),
            "jobs_this_month": TailoringJob.query.filter(TailoringJob.created_at >= tw["month"]).count(),
            "avg_duration_seconds": round(float(avg_duration or 0), 1),
            "completed_count": completed_count,
        }
    )


@admin_bp.route("/admin/api/users")
@_admin_required
def admin_users():
    results = (
        db.session.query(User, func.count(TailoringJob.id).label("jobs_count"))
        .outerjoin(TailoringJob, User.id == TailoringJob.user_id)
        .group_by(User.id)
        .order_by(User.created_at.desc())
        .all()
    )
    return jsonify(
        {
            "total": len(results),
            "users": [
                {
                    "id": u.id,
                    "email": u.email,
                    "name": u.name,
                    "picture": u.picture_url,
                    "is_admin": u.is_admin,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "last_login": u.last_login_at.isoformat() if u.last_login_at else None,
                    "jobs_count": jobs_count,
                }
                for u, jobs_count in results
            ],
        }
    )


@admin_bp.route("/admin/api/user-jobs/<user_id>")
@_admin_required
def admin_user_jobs(user_id: str):
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    jobs_list = TailoringJob.query.filter_by(user_id=user_id).order_by(TailoringJob.created_at.desc()).all()
    return jsonify(
        {
            "user": {"id": user.id, "email": user.email, "name": user.name},
            "jobs": [
                {
                    "id": j.id,
                    "status": j.status,
                    "job_title": j.job_title,
                    "company": j.company,
                    "match_score": j.match_score,
                    "original_match_score": j.original_match_score,
                    "rewrite_mode": j.rewrite_mode,
                    "model_used": j.model_used,
                    "job_description_snippet": j.job_description_snippet,
                    "job_description_full": j.job_description_full,
                    "original_resume_text": j.original_resume_text,
                    "ats_resume_md": j.ats_resume_md,
                    "talking_points_md": j.talking_points_md,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "duration_seconds": j.duration_seconds,
                    "error_message": j.error_message,
                }
                for j in jobs_list
            ],
        }
    )


@admin_bp.route("/admin/api/live-stats")
@_admin_required
def admin_live_stats():
    rusage = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
    if platform_mod.system() == "Darwin":
        mem_mb = rusage.ru_maxrss / (1024 * 1024)
    else:
        mem_mb = rusage.ru_maxrss / 1024

    with pipeline_errors_lock:
        recent_errors_count = len(pipeline_errors)

    live = _live_pipeline_state()
    return jsonify(
        {
            **live,
            "max_queue_depth": live["max_queue"],
            "memory_mb": round(mem_mb, 1),
            "thread_count": threading.active_count(),
            "usage_stats": usage_tracker.get_stats(),
            "analytics_stats": pipeline_analytics.get_global_stats(),
            "recent_errors_count": recent_errors_count,
        }
    )


@admin_bp.route("/admin/api/stats")
@_admin_required
def admin_stats():
    """Comprehensive stats: jobs by status, time trends, success rate, saved resumes."""
    tw = _time_windows()
    status = _jobs_by_status()

    jobs_today = TailoringJob.query.filter(TailoringJob.created_at >= tw["today"]).count()
    jobs_this_week = TailoringJob.query.filter(TailoringJob.created_at >= tw["week"]).count()
    jobs_this_month = TailoringJob.query.filter(TailoringJob.created_at >= tw["month"]).count()

    completed = status.get("complete", 0)
    errors = status.get("error", 0)
    total_finished = completed + errors
    success_rate = (completed / total_finished * 100) if total_finished > 0 else 100.0

    # Match score improvement (avg before vs after)
    score_improvement = db.session.query(
        func.avg(
            case(
                (
                    (TailoringJob.match_score.isnot(None)) & (TailoringJob.original_match_score.isnot(None)),
                    TailoringJob.match_score - TailoringJob.original_match_score,
                ),
                else_=None,
            )
        )
    ).scalar()
    avg_improvement = round(float(score_improvement or 0), 1)

    # Saved resumes count
    saved_resumes_count = SavedResume.query.count()

    total_jobs = sum(status.values())
    total_users = User.query.count()

    return jsonify(
        {
            "total_jobs": total_jobs,
            "total_users": total_users,
            "jobs_by_status": status,
            "jobs_today": jobs_today,
            "jobs_this_week": jobs_this_week,
            "jobs_this_month": jobs_this_month,
            "success_rate": round(success_rate, 1),
            "avg_match_improvement": avg_improvement,
            "saved_resumes_count": saved_resumes_count,
        }
    )


@admin_bp.route("/admin/api/recent-jobs")
@_admin_required
def admin_recent_jobs():
    """Last 20 jobs across all users (for dashboard activity)."""
    jobs = TailoringJob.query.order_by(TailoringJob.created_at.desc()).limit(20).all()
    user_ids = [j.user_id for j in jobs if j.user_id]
    users_by_id = {u.id: u for u in User.query.filter(User.id.in_(user_ids)).all()} if user_ids else {}

    return jsonify(
        {
            "jobs": [
                {
                    "id": j.id,
                    "status": j.status,
                    "job_title": j.job_title,
                    "company": j.company,
                    "match_score": j.match_score,
                    "user_email": (users_by_id[j.user_id].email if j.user_id in users_by_id else "Anonymous"),
                    "user_name": (users_by_id[j.user_id].name if j.user_id in users_by_id else "—"),
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "duration_seconds": j.duration_seconds,
                }
                for j in jobs
            ],
        }
    )


@admin_bp.route("/admin/api/errors/clear", methods=["POST"])
@_admin_required
def admin_clear_errors():
    """Clear the pipeline error log."""
    with pipeline_errors_lock:
        count = len(pipeline_errors)
        pipeline_errors.clear()
    track("admin.errors.cleared", category="admin", metadata={"count": count})
    return jsonify({"ok": True})


@admin_bp.route("/admin/api/diagnostics")
@_admin_required
def admin_diagnostics():
    """Run health checks and return diagnostic info for troubleshooting."""
    import sys

    from app.services.cache import is_available as redis_available
    from storage import r2_storage

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": "healthy",
        "checks": {},
        "environment": {},
        "recommendations": [],
    }

    # Database
    try:
        db.session.execute(text("SELECT 1"))
        db_uri = str(db.engine.url)
        if "@" in db_uri:
            db_uri = db_uri.split("@")[-1]  # Hide credentials
        results["checks"]["database"] = {
            "status": "healthy",
            "message": "Connected",
            "type": "postgresql" if "postgresql" in str(db.engine.url) else "sqlite",
            "detail": db_uri[:80] + "..." if len(db_uri) > 80 else db_uri,
        }
    except Exception as e:
        results["checks"]["database"] = {
            "status": "unhealthy",
            "message": str(e),
            "detail": None,
        }
        results["overall"] = "degraded"
        results["recommendations"].append("Check DATABASE_URL and database connectivity.")

    # OpenRouter API key
    config = AdminConfigManager.load()
    has_key = bool(config.api_key and config.api_key.strip())
    if has_key:
        try:
            r = http_requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {config.api_key.strip()}"},
                timeout=8,
            )
            if r.status_code == 200:
                results["checks"]["openrouter"] = {
                    "status": "healthy",
                    "message": "API key valid",
                    "detail": f"HTTP {r.status_code}",
                }
            else:
                results["checks"]["openrouter"] = {
                    "status": "unhealthy",
                    "message": f"HTTP {r.status_code}",
                    "detail": r.text[:200] if r.text else None,
                }
                results["overall"] = "degraded"
                results["recommendations"].append("Verify OpenRouter API key at openrouter.ai/keys")
        except Exception as e:
            results["checks"]["openrouter"] = {
                "status": "unhealthy",
                "message": str(e),
                "detail": None,
            }
            results["overall"] = "degraded"
            results["recommendations"].append("Check network connectivity to openrouter.ai")
    else:
        results["checks"]["openrouter"] = {
            "status": "unhealthy",
            "message": "No API key configured",
            "detail": "Set API key in Configuration tab",
        }
        results["overall"] = "degraded"
        results["recommendations"].append("Configure OpenRouter API key in Configuration tab.")

    # R2 storage
    if r2_storage.is_configured:
        try:
            r2_storage._client.list_objects_v2(Bucket=r2_storage._bucket, MaxKeys=1)
            results["checks"]["r2_storage"] = {
                "status": "healthy",
                "message": "Connected",
                "detail": f"Bucket: {r2_storage._bucket}",
            }
        except Exception as e:
            results["checks"]["r2_storage"] = {
                "status": "unhealthy",
                "message": str(e),
                "detail": None,
            }
            if results["overall"] == "healthy":
                results["overall"] = "degraded"
            results["recommendations"].append("Check R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.")
    else:
        results["checks"]["r2_storage"] = {
            "status": "skipped",
            "message": "Not configured",
            "detail": "Using local storage (files stored on disk)",
        }

    # Redis
    if redis_available():
        try:
            from app.services.cache import get_redis

            client = get_redis()
            if client:
                client.ping()
                results["checks"]["redis"] = {
                    "status": "healthy",
                    "message": "Connected",
                    "detail": "Rate limiting and cache active",
                }
            else:
                results["checks"]["redis"] = {"status": "skipped", "message": "N/A", "detail": None}
        except Exception as e:
            results["checks"]["redis"] = {
                "status": "unhealthy",
                "message": str(e),
                "detail": None,
            }
    else:
        results["checks"]["redis"] = {
            "status": "skipped",
            "message": "Not configured",
            "detail": "Using in-memory rate limiting",
        }

    # Environment (no secrets)
    results["environment"] = {
        "python_version": sys.version.split()[0],
        "flask_env": os.environ.get("FLASK_ENV", "production"),
        "database_type": "postgresql" if "postgresql" in str(db.engine.url) else "sqlite",
        "r2_configured": r2_storage.is_configured,
        "redis_configured": redis_available(),
        "admin_password_set": AdminConfigManager.has_password(),
    }

    # Pipeline health
    with pipeline_errors_lock:
        err_count = len(pipeline_errors)
    results["checks"]["pipeline"] = {
        "status": "healthy" if err_count < 10 else "warning",
        "message": f"{err_count} recent errors in log",
        "detail": "Check Errors tab for details" if err_count > 0 else "No recent errors",
    }
    if err_count >= 10 and results["overall"] == "healthy":
        results["overall"] = "degraded"

    return jsonify(results)


@admin_bp.route("/admin/api/users/<user_id>/admin", methods=["POST"])
@_admin_required
def admin_set_user_admin(user_id: str):
    """Promote or demote a user to/from admin."""
    data = request.get_json(force=True) or {}
    is_admin = bool(data.get("is_admin", False)) if data else False

    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user.is_admin = is_admin
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to update user admin status")
        return jsonify({"error": f"Failed to update: {e}"}), 500
    action = "admin.user.promoted" if is_admin else "admin.user.demoted"
    track(action, category="admin", user_id=user_id)
    return jsonify({"ok": True, "user_id": user_id, "is_admin": is_admin})


# ═══════════════════════════════════════════════════════════════════════════════
# Observability Hub Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@admin_bp.route("/admin/api/observability/product-usage")
@_admin_required
def admin_product_usage():
    """Product usage metrics: DAU/WAU/MAU, feature adoption, funnel data."""
    now = datetime.now(timezone.utc)

    # Active users by window
    dau = (
        db.session.query(func.count(func.distinct(TailoringJob.user_id)))
        .filter(TailoringJob.created_at >= now.replace(hour=0, minute=0, second=0))
        .scalar()
        or 0
    )
    wau = (
        db.session.query(func.count(func.distinct(TailoringJob.user_id)))
        .filter(TailoringJob.created_at >= now - timedelta(days=7))
        .scalar()
        or 0
    )
    mau = (
        db.session.query(func.count(func.distinct(TailoringJob.user_id)))
        .filter(TailoringJob.created_at >= now - timedelta(days=30))
        .scalar()
        or 0
    )

    # New user registrations over 30 days
    new_users_30d = User.query.filter(User.created_at >= now - timedelta(days=30)).count()

    # Feature adoption: saved resumes, tracker apps, downloads
    users_with_saved_resumes = db.session.query(func.count(func.distinct(SavedResume.user_id))).scalar() or 0
    users_with_tracker = db.session.query(func.count(func.distinct(JobApplication.user_id))).scalar() or 0
    total_users = User.query.count()

    # Jobs per day (last 30 days) for trend chart
    jobs_per_day = (
        db.session.query(
            func.date(TailoringJob.created_at).label("day"),
            func.count(TailoringJob.id).label("total"),
            func.count(case((TailoringJob.status == "complete", 1))).label("completed"),
            func.count(case((TailoringJob.status == "error", 1))).label("failed"),
        )
        .filter(TailoringJob.created_at >= now - timedelta(days=30))
        .group_by(func.date(TailoringJob.created_at))
        .order_by(func.date(TailoringJob.created_at))
        .all()
    )

    # Users per day (login activity)
    users_per_day = (
        db.session.query(
            func.date(User.last_login_at).label("day"),
            func.count(User.id).label("count"),
        )
        .filter(User.last_login_at >= now - timedelta(days=30))
        .group_by(func.date(User.last_login_at))
        .order_by(func.date(User.last_login_at))
        .all()
    )

    # Tracker funnel (applications by status)
    tracker_by_status = dict(
        db.session.query(JobApplication.status, func.count(JobApplication.id))
        .group_by(JobApplication.status)
        .all()
    )

    return jsonify(
        {
            "dau": dau,
            "wau": wau,
            "mau": mau,
            "total_users": total_users,
            "new_users_30d": new_users_30d,
            "feature_adoption": {
                "saved_resumes_users": users_with_saved_resumes,
                "tracker_users": users_with_tracker,
                "total_users": total_users,
                "saved_resumes_pct": round(users_with_saved_resumes / max(total_users, 1) * 100, 1),
                "tracker_pct": round(users_with_tracker / max(total_users, 1) * 100, 1),
            },
            "jobs_per_day": [
                {"date": str(d).split()[0] if d else "1970-01-01", "total": t, "completed": c, "failed": f}
                for d, t, c, f in jobs_per_day
            ],
            "users_per_day": [
                {"date": str(d).split()[0] if d else "1970-01-01", "count": c} for d, c in users_per_day
            ],
            "tracker_funnel": tracker_by_status,
        }
    )


@admin_bp.route("/admin/api/observability/reliability")
@_admin_required
def admin_reliability():
    """Pipeline reliability metrics: success/failure rates, durations, errors by model."""
    now = datetime.now(timezone.utc)
    month_ago = now - timedelta(days=30)

    status_30d = _jobs_by_status(since=month_ago)
    completed = status_30d.get("complete", 0)
    errors = status_30d.get("error", 0)
    total_finished = completed + errors
    success_rate = round(completed / max(total_finished, 1) * 100, 1)

    # Duration percentiles (completed jobs, last 30 days)
    durations = [
        d[0]
        for d in db.session.query(TailoringJob.duration_seconds)
        .filter(
            TailoringJob.created_at >= month_ago,
            TailoringJob.duration_seconds.isnot(None),
            TailoringJob.status == "complete",
        )
        .order_by(TailoringJob.duration_seconds)
        .all()
    ]
    p50 = durations[len(durations) // 2] if durations else 0
    p95 = durations[int(len(durations) * 0.95)] if durations else 0
    avg_dur = round(sum(durations) / max(len(durations), 1), 1)

    # Errors by model (last 30 days)
    errors_by_model = dict(
        db.session.query(TailoringJob.model_used, func.count(TailoringJob.id))
        .filter(
            TailoringJob.created_at >= month_ago,
            TailoringJob.status == "error",
        )
        .group_by(TailoringJob.model_used)
        .all()
    )

    # Failure rate by day
    failures_per_day = (
        db.session.query(
            func.date(TailoringJob.created_at).label("day"),
            func.count(TailoringJob.id).label("total"),
            func.count(case((TailoringJob.status == "error", 1))).label("errors"),
        )
        .filter(TailoringJob.created_at >= month_ago)
        .group_by(func.date(TailoringJob.created_at))
        .order_by(func.date(TailoringJob.created_at))
        .all()
    )

    # Recent errors (from DB, durable)
    recent_errors_db = (
        db.session.query(
            TailoringJob.created_at,
            TailoringJob.model_used,
            TailoringJob.error_message,
        )
        .filter(TailoringJob.status == "error")
        .order_by(TailoringJob.created_at.desc())
        .limit(25)
        .all()
    )

    return jsonify(
        {
            "success_rate": success_rate,
            "total_completed_30d": completed,
            "total_errors_30d": errors,
            "duration_p50": round(p50, 1),
            "duration_p95": round(p95, 1),
            "duration_avg": avg_dur,
            "errors_by_model": errors_by_model,
            "failures_per_day": [
                {"date": str(d).split()[0] if d else "1970-01-01", "total": t, "errors": e}
                for d, t, e in failures_per_day
            ],
            "recent_errors": [
                {
                    "time": t.isoformat() if t else None,
                    "model": m,
                    "error": (msg or "")[:300],
                }
                for t, m, msg in recent_errors_db
            ],
            "live": _live_pipeline_state(),
        }
    )


@admin_bp.route("/admin/api/observability/cost")
@_admin_required
def admin_cost():
    """Cost and model economics: spend by day/model, cost per resume, anomalies."""
    now = datetime.now(timezone.utc)

    # In-memory global stats (current session)
    live_stats = pipeline_analytics.get_global_stats()

    # DB-backed: jobs by model (all time + last 30 days)
    model_usage = (
        db.session.query(
            TailoringJob.model_used,
            func.count(TailoringJob.id).label("count"),
        )
        .filter(TailoringJob.model_used.isnot(None))
        .group_by(TailoringJob.model_used)
        .all()
    )
    model_usage_30d = (
        db.session.query(
            TailoringJob.model_used,
            func.count(TailoringJob.id).label("count"),
        )
        .filter(
            TailoringJob.created_at >= now - timedelta(days=30),
            TailoringJob.model_used.isnot(None),
        )
        .group_by(TailoringJob.model_used)
        .all()
    )

    # Estimated cost from events (jobs completed with cost data)
    cost_events = (
        db.session.query(AnalyticsEvent.metadata_json)
        .filter(
            AnalyticsEvent.event_name == "tailor.job.completed",
            AnalyticsEvent.event_time >= now - timedelta(days=30),
        )
        .all()
    )

    model_cost: dict[str, float] = {}
    total_tokens_30d = 0
    total_cost_30d = 0.0
    for (meta,) in cost_events:
        if not meta:
            continue
        cost = meta.get("estimated_cost_usd", 0) or 0
        tokens = meta.get("total_tokens", 0) or 0
        model_name = meta.get("model", "unknown")
        total_cost_30d += cost
        total_tokens_30d += tokens
        model_cost[model_name] = model_cost.get(model_name, 0) + cost

    # Jobs per day for cost estimation
    jobs_per_day_cost = (
        db.session.query(
            func.date(TailoringJob.created_at).label("day"),
            func.count(TailoringJob.id).label("count"),
        )
        .filter(
            TailoringJob.created_at >= now - timedelta(days=30),
            TailoringJob.status == "complete",
        )
        .group_by(func.date(TailoringJob.created_at))
        .order_by(func.date(TailoringJob.created_at))
        .all()
    )

    completed_30d = sum(c for _, c in jobs_per_day_cost)
    cost_per_resume = round(total_cost_30d / max(completed_30d, 1), 4)

    return jsonify(
        {
            "live_stats": {
                "total_tokens_session": live_stats.get("total_tokens", 0),
                "total_cost_session": round(live_stats.get("total_cost_usd", 0), 4),
                "total_jobs_session": live_stats.get("total_jobs", 0),
                "avg_cost_per_job": live_stats.get("avg_cost_per_job", 0),
            },
            "cost_30d": {
                "total_cost_usd": round(total_cost_30d, 4),
                "total_tokens": total_tokens_30d,
                "completed_jobs": completed_30d,
                "cost_per_resume": cost_per_resume,
            },
            "model_usage_all": {m: c for m, c in model_usage},
            "model_usage_30d": {m: c for m, c in model_usage_30d},
            "model_cost_30d": {k: round(v, 4) for k, v in model_cost.items()},
            "jobs_per_day": [
                {"date": str(d).split()[0] if d else "1970-01-01", "count": c} for d, c in jobs_per_day_cost
            ],
        }
    )


@admin_bp.route("/admin/api/observability/audit")
@_admin_required
def admin_audit():
    """Admin/security audit trail: admin actions, auth events, suspicious activity."""
    now = datetime.now(timezone.utc)
    days = int(request.args.get("days", 30))
    since = now - timedelta(days=days)

    # Admin actions from analytics events
    admin_events = (
        AnalyticsEvent.query.filter(
            AnalyticsEvent.category == "admin",
            AnalyticsEvent.event_time >= since,
        )
        .order_by(AnalyticsEvent.event_time.desc())
        .limit(100)
        .all()
    )

    # Auth events (failures, resets)
    auth_events = (
        AnalyticsEvent.query.filter(
            AnalyticsEvent.category == "auth",
            AnalyticsEvent.event_time >= since,
        )
        .order_by(AnalyticsEvent.event_time.desc())
        .limit(100)
        .all()
    )

    # Auth failure counts by day (for spike detection)
    auth_failures_by_day = (
        db.session.query(
            func.date(AnalyticsEvent.event_time).label("day"),
            func.count(AnalyticsEvent.id).label("count"),
        )
        .filter(
            AnalyticsEvent.event_name.in_(["auth.login.failed", "admin.login.failed"]),
            AnalyticsEvent.event_time >= since,
        )
        .group_by(func.date(AnalyticsEvent.event_time))
        .order_by(func.date(AnalyticsEvent.event_time))
        .all()
    )

    # High-volume users (potential abuse)
    heavy_users = (
        db.session.query(
            TailoringJob.user_id,
            func.count(TailoringJob.id).label("job_count"),
        )
        .filter(TailoringJob.created_at >= since)
        .group_by(TailoringJob.user_id)
        .having(func.count(TailoringJob.id) > 10)
        .order_by(func.count(TailoringJob.id).desc())
        .limit(20)
        .all()
    )
    heavy_user_details = []
    if heavy_users:
        user_ids = [uid for uid, _ in heavy_users if uid]
        users_map = {u.id: u for u in User.query.filter(User.id.in_(user_ids)).all()} if user_ids else {}
        for uid, count in heavy_users:
            u = users_map.get(uid)
            heavy_user_details.append(
                {
                    "user_id": uid,
                    "email": u.email if u else "—",
                    "name": u.name if u else "—",
                    "job_count": count,
                }
            )

    return jsonify(
        {
            "admin_actions": [
                {
                    "event": e.event_name,
                    "time": e.event_time.isoformat() if e.event_time else None,
                    "metadata": e.metadata_json,
                }
                for e in admin_events
            ],
            "auth_events": [
                {
                    "event": e.event_name,
                    "time": e.event_time.isoformat() if e.event_time else None,
                    "metadata": e.metadata_json,
                }
                for e in auth_events
            ],
            "auth_failures_by_day": [
                {"date": str(d).split()[0] if d else "1970-01-01", "count": c}
                for d, c in auth_failures_by_day
            ],
            "heavy_users": heavy_user_details,
        }
    )


@admin_bp.route("/admin/api/observability/events")
@_admin_required
def admin_events_feed():
    """Live event feed — most recent analytics events (paginated)."""
    page = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 50)), 100)
    category = request.args.get("category")
    event_name = request.args.get("event_name")

    query = AnalyticsEvent.query

    if category:
        query = query.filter(AnalyticsEvent.category == category)
    if event_name:
        query = query.filter(AnalyticsEvent.event_name == event_name)

    total = query.count()
    events = (
        query.order_by(AnalyticsEvent.event_time.desc()).offset((page - 1) * per_page).limit(per_page).all()
    )

    return jsonify(
        {
            "total": total,
            "page": page,
            "per_page": per_page,
            "events": [
                {
                    "id": e.id,
                    "event_name": e.event_name,
                    "category": e.category,
                    "time": e.event_time.isoformat() if e.event_time else None,
                    "job_id": e.job_id,
                    "request_id": e.request_id,
                    "metadata": e.metadata_json,
                }
                for e in events
            ],
        }
    )


@admin_bp.route("/admin/api/observability/alerts")
@_admin_required
def admin_alerts():
    """Check SLO breaches and generate alerts for the admin dashboard."""
    now = datetime.now(timezone.utc)
    alerts = []

    # 1. Error rate spike: >20% failure in last 24h
    status_24h = _jobs_by_status(since=now - timedelta(hours=24))
    completed_24h = status_24h.get("complete", 0)
    errors_24h = status_24h.get("error", 0)
    total_24h = completed_24h + errors_24h
    if total_24h >= 3 and errors_24h / total_24h > 0.2:
        alerts.append(
            {
                "severity": "high",
                "type": "error_rate_spike",
                "message": f"High error rate: {errors_24h}/{total_24h} jobs failed in 24h ({round(errors_24h / total_24h * 100)}%)",
            }
        )

    # 2. Queue saturation: queue depth > 80% of max
    live = _live_pipeline_state()
    current_queue = live["queue_depth"]
    if current_queue > MAX_QUEUE_DEPTH * 0.8:
        alerts.append(
            {
                "severity": "high",
                "type": "queue_saturation",
                "message": f"Queue near capacity: {current_queue}/{MAX_QUEUE_DEPTH}",
            }
        )

    # 3. No completed jobs in 24h (if there were attempts)
    running_24h = status_24h.get("running", 0)
    if total_24h == 0 and running_24h > 0:
        alerts.append(
            {
                "severity": "medium",
                "type": "pipeline_stalled",
                "message": f"{running_24h} jobs running but none completed in 24h",
            }
        )

    # 4. Auth failure burst: >10 failures in last hour
    auth_failures_1h = AnalyticsEvent.query.filter(
        AnalyticsEvent.event_name.in_(["auth.login.failed", "admin.login.failed"]),
        AnalyticsEvent.event_time >= now - timedelta(hours=1),
    ).count()
    if auth_failures_1h > 10:
        alerts.append(
            {
                "severity": "medium",
                "type": "auth_failure_burst",
                "message": f"{auth_failures_1h} auth failures in the last hour",
            }
        )

    # 5. Cost anomaly: session cost > $5 (unusual for GPT-4o-mini)
    live_cost = pipeline_analytics.get_global_stats().get("total_cost_usd", 0)
    if live_cost > 5.0:
        alerts.append(
            {
                "severity": "medium",
                "type": "cost_anomaly",
                "message": f"Session cost unusually high: ${live_cost:.2f}",
            }
        )

    return jsonify(
        {
            "alerts": alerts,
            "checked_at": now.isoformat(),
            "slo": {
                "error_rate_24h": round(errors_24h / max(total_24h, 1) * 100, 1),
                "queue_utilization": round(current_queue / max(MAX_QUEUE_DEPTH, 1) * 100, 1),
                "auth_failures_1h": auth_failures_1h,
                "session_cost_usd": round(live_cost, 4),
            },
        }
    )


@admin_bp.route("/admin/api/observability/retention", methods=["POST"])
@_admin_required
def admin_run_retention():
    """Prune old analytics events beyond retention period."""
    days = int(request.get_json(force=True).get("days", 90))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = AnalyticsEvent.query.filter(AnalyticsEvent.event_time < cutoff).delete()
    db.session.commit()
    track("admin.retention.executed", category="admin", metadata={"days": days, "deleted_count": deleted})
    return jsonify({"ok": True, "deleted": deleted, "cutoff": cutoff.isoformat()})
