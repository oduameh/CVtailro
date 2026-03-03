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
from app.models import SavedResume, TailoringJob, User
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
from app.services.usage import login_rate_limiter, usage_tracker

logger = logging.getLogger("cvtailro.admin")

admin_bp = Blueprint("admin", __name__)
csrf.exempt(admin_bp)  # Admin uses session-based auth


def _admin_required(f):
    """Decorator checking session-based admin auth."""
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_authenticated"):
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)

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
        config = AdminConfigManager.load()
        config.admin_password_hash = AdminConfigManager._hash_password(password)
        AdminConfigManager.save(config)
        session["admin_authenticated"] = True
        login_rate_limiter.reset(client_ip)
        return jsonify({"ok": True, "message": "Password set successfully"})

    if AdminConfigManager.verify_password(password):
        session["admin_authenticated"] = True
        login_rate_limiter.reset(client_ip)
        return jsonify({"ok": True})

    login_rate_limiter.record_failure(client_ip)
    return jsonify({"error": "Invalid password"}), 401


@admin_bp.route("/admin/api/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_authenticated", None)
    return jsonify({"ok": True})


@admin_bp.route("/admin/api/config", methods=["GET"])
@_admin_required
def admin_get_config():
    config = AdminConfigManager.load()
    masked_key = ""
    if config.api_key:
        masked_key = config.api_key[:8] + "..." if len(config.api_key) > 8 else config.api_key
    return jsonify(
        {
            "api_key": masked_key,
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
        # Only update the key if user entered a real key (not the masked placeholder)
        if new_key and "..." not in new_key:
            config.api_key = new_key
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
    return jsonify({"ok": True, "updated_at": config.updated_at})


@admin_bp.route("/admin/api/test-key", methods=["POST"])
@_admin_required
def admin_test_key():
    data = request.get_json(force=True)
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})
    try:
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
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # In-memory pipeline stats (resets on restart)
    pipeline_stats = pipeline_analytics.get_global_stats()

    # DB-backed: jobs per day for last 30 days (for charts)
    jobs_per_day = (
        db.session.query(
            func.date(TailoringJob.created_at).label("day"),
            func.count(TailoringJob.id).label("count"),
        )
        .filter(TailoringJob.created_at >= today_start - timedelta(days=30))
        .group_by(func.date(TailoringJob.created_at))
        .order_by(func.date(TailoringJob.created_at))
        .all()
    )
    jobs_over_time = [{"date": str(d).split()[0] if d else "1970-01-01", "jobs": c} for d, c in jobs_per_day]

    # DB-backed: jobs by status for pie chart
    status_counts = (
        db.session.query(TailoringJob.status, func.count(TailoringJob.id)).group_by(TailoringJob.status).all()
    )
    jobs_by_status = {s: c for s, c in status_counts}

    # DB-backed: avg duration, completed jobs
    completed_count = jobs_by_status.get("completed", 0)
    avg_duration = (
        db.session.query(func.avg(TailoringJob.duration_seconds))
        .filter(TailoringJob.status == "completed", TailoringJob.duration_seconds.isnot(None))
        .scalar()
    )
    avg_duration_seconds = round(float(avg_duration or 0), 1)

    return jsonify(
        {
            **pipeline_stats,
            "jobs_over_time": jobs_over_time,
            "jobs_by_status": jobs_by_status,
            "jobs_today": TailoringJob.query.filter(TailoringJob.created_at >= today_start).count(),
            "jobs_this_week": TailoringJob.query.filter(
                TailoringJob.created_at >= today_start - timedelta(days=7)
            ).count(),
            "jobs_this_month": TailoringJob.query.filter(
                TailoringJob.created_at >= today_start - timedelta(days=30)
            ).count(),
            "avg_duration_seconds": avg_duration_seconds,
            "completed_count": completed_count,
        }
    )


@admin_bp.route("/admin/api/users")
def admin_users():
    if not session.get("admin_authenticated") and not (
        current_user.is_authenticated and current_user.is_admin
    ):
        return jsonify({"error": "Not authenticated"}), 401
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
def admin_user_jobs(user_id: str):
    if not session.get("admin_authenticated") and not (
        current_user.is_authenticated and current_user.is_admin
    ):
        return jsonify({"error": "Not authenticated"}), 401
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
    try:
        available_slots = pipeline_semaphore._value
        active_pipelines = MAX_CONCURRENT_PIPELINES - available_slots
    except AttributeError:
        active_pipelines = -1

    with pipeline_queue_lock:
        current_queue_depth = pipeline_queue_depth

    rusage = resource_mod.getrusage(resource_mod.RUSAGE_SELF)
    if platform_mod.system() == "Darwin":
        mem_mb = rusage.ru_maxrss / (1024 * 1024)
    else:
        mem_mb = rusage.ru_maxrss / 1024

    with pipeline_errors_lock:
        recent_errors_count = len(pipeline_errors)

    return jsonify(
        {
            "active_pipelines": active_pipelines,
            "queue_depth": current_queue_depth,
            "max_concurrent": MAX_CONCURRENT_PIPELINES,
            "max_queue_depth": MAX_QUEUE_DEPTH,
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
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)

    # Jobs by status
    status_counts = (
        db.session.query(TailoringJob.status, func.count(TailoringJob.id)).group_by(TailoringJob.status).all()
    )
    jobs_by_status = {s: c for s, c in status_counts}

    # Time-based counts
    jobs_today = TailoringJob.query.filter(TailoringJob.created_at >= today_start).count()
    jobs_this_week = TailoringJob.query.filter(TailoringJob.created_at >= week_start).count()
    jobs_this_month = TailoringJob.query.filter(TailoringJob.created_at >= month_start).count()

    # Success rate (completed jobs with resume)
    completed = jobs_by_status.get("completed", 0)
    errors = jobs_by_status.get("error", 0)
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

    total_jobs = sum(jobs_by_status.values())
    total_users = User.query.count()

    return jsonify(
        {
            "total_jobs": total_jobs,
            "total_users": total_users,
            "jobs_by_status": jobs_by_status,
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
        pipeline_errors.clear()
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
    return jsonify({"ok": True, "user_id": user_id, "is_admin": is_admin})
