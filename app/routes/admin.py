"""Admin panel routes — configuration, analytics, user management, system stats."""

from __future__ import annotations

import logging
import platform as platform_mod
import resource as resource_mod
import threading
from datetime import datetime, timedelta, timezone

import requests as http_requests
from flask import Blueprint, jsonify, render_template, request, session
from flask_login import current_user
from sqlalchemy import func, case

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
    return jsonify({
        "api_key": masked_key,
        "default_model": config.default_model,
        "allow_user_model_selection": config.allow_user_model_selection,
        "rate_limit_per_hour": config.rate_limit_per_hour,
        "updated_at": config.updated_at,
    })


@admin_bp.route("/admin/api/config", methods=["POST"])
@_admin_required
def admin_save_config():
    data = request.get_json(force=True)
    config = AdminConfigManager.load()

    if "api_key" in data:
        config.api_key = data["api_key"]
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
        return jsonify({"valid": r.status_code == 200} | ({"error": f"HTTP {r.status_code}"} if r.status_code != 200 else {}))
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
    return jsonify(pipeline_analytics.get_global_stats())


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
    return jsonify({
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
    })


@admin_bp.route("/admin/api/user-jobs/<user_id>")
def admin_user_jobs(user_id: str):
    if not session.get("admin_authenticated") and not (
        current_user.is_authenticated and current_user.is_admin
    ):
        return jsonify({"error": "Not authenticated"}), 401
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    jobs_list = (
        TailoringJob.query.filter_by(user_id=user_id)
        .order_by(TailoringJob.created_at.desc())
        .all()
    )
    return jsonify({
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
    })


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

    return jsonify({
        "active_pipelines": active_pipelines,
        "queue_depth": current_queue_depth,
        "max_concurrent": MAX_CONCURRENT_PIPELINES,
        "max_queue_depth": MAX_QUEUE_DEPTH,
        "memory_mb": round(mem_mb, 1),
        "thread_count": threading.active_count(),
        "usage_stats": usage_tracker.get_stats(),
        "analytics_stats": pipeline_analytics.get_global_stats(),
        "recent_errors_count": recent_errors_count,
    })


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
        db.session.query(TailoringJob.status, func.count(TailoringJob.id))
        .group_by(TailoringJob.status)
        .all()
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
    score_improvement = (
        db.session.query(
            func.avg(
                case(
                    (
                        (TailoringJob.match_score.isnot(None))
                        & (TailoringJob.original_match_score.isnot(None)),
                        TailoringJob.match_score - TailoringJob.original_match_score,
                    ),
                    else_=None,
                )
            )
        ).scalar()
    )
    avg_improvement = round(float(score_improvement or 0), 1)

    # Saved resumes count
    saved_resumes_count = SavedResume.query.count()

    total_jobs = sum(jobs_by_status.values())

    return jsonify({
        "total_jobs": total_jobs,
        "jobs_by_status": jobs_by_status,
        "jobs_today": jobs_today,
        "jobs_this_week": jobs_this_week,
        "jobs_this_month": jobs_this_month,
        "success_rate": round(success_rate, 1),
        "avg_match_improvement": avg_improvement,
        "saved_resumes_count": saved_resumes_count,
    })


@admin_bp.route("/admin/api/recent-jobs")
@_admin_required
def admin_recent_jobs():
    """Last 20 jobs across all users (for dashboard activity)."""
    jobs = (
        TailoringJob.query
        .order_by(TailoringJob.created_at.desc())
        .limit(20)
        .all()
    )
    user_ids = [j.user_id for j in jobs if j.user_id]
    users_by_id = {u.id: u for u in User.query.filter(User.id.in_(user_ids)).all()} if user_ids else {}

    return jsonify({
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
    })


@admin_bp.route("/admin/api/errors/clear", methods=["POST"])
@_admin_required
def admin_clear_errors():
    """Clear the pipeline error log."""
    with pipeline_errors_lock:
        pipeline_errors.clear()
    return jsonify({"ok": True})


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
    db.session.commit()
    return jsonify({"ok": True, "user_id": user_id, "is_admin": is_admin})
