"""Job Application Tracker routes — CRUD for tracking applications."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from app.extensions import db
from app.models import JobApplication
from app.services.telemetry import track

tracker_bp = Blueprint("tracker", __name__)


def _parse_date(value: str | None):
    """Parse ISO date string, returning None on invalid input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@tracker_bp.route("/api/tracker", methods=["GET"])
@login_required
def list_applications():
    apps = (
        JobApplication.query.filter_by(user_id=current_user.id)
        .order_by(JobApplication.updated_at.desc())
        .all()
    )
    return jsonify(
        [
            {
                "id": a.id,
                "company": a.company,
                "job_title": a.job_title,
                "status": a.status,
                "url": a.url,
                "notes": a.notes,
                "applied_date": a.applied_date.isoformat() if a.applied_date else None,
                "interview_date": a.interview_date.isoformat() if a.interview_date else None,
                "tailoring_job_id": a.tailoring_job_id,
                "created_at": a.created_at.isoformat(),
                "updated_at": a.updated_at.isoformat(),
            }
            for a in apps
        ]
    )


@tracker_bp.route("/api/tracker", methods=["POST"])
@login_required
def create_application():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    company = (data.get("company") or "").strip()
    job_title = (data.get("job_title") or "").strip()
    if not company or not job_title:
        return jsonify({"error": "Company and job title are required"}), 400

    app = JobApplication(
        user_id=current_user.id,
        company=company,
        job_title=job_title,
        status=data.get("status", "saved"),
        url=data.get("url"),
        notes=data.get("notes"),
        tailoring_job_id=data.get("tailoring_job_id"),
        applied_date=_parse_date(data.get("applied_date")),
    )
    db.session.add(app)
    db.session.commit()
    track("tracker.application.created", category="feature", user_id=current_user.id,
          metadata={"status": app.status, "has_tailoring_job": bool(app.tailoring_job_id)})
    return jsonify({"id": app.id, "status": "created"}), 201


@tracker_bp.route("/api/tracker/<app_id>", methods=["PUT"])
@login_required
def update_application(app_id):
    app = JobApplication.query.filter_by(id=app_id, user_id=current_user.id).first()
    if not app:
        return jsonify({"error": "Application not found"}), 404

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    for field in ("company", "job_title", "status", "url", "notes"):
        if field in data:
            setattr(app, field, data[field])
    if "applied_date" in data:
        app.applied_date = _parse_date(data.get("applied_date"))
    if "interview_date" in data:
        app.interview_date = _parse_date(data.get("interview_date"))

    app.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    track("tracker.application.updated", category="feature", user_id=current_user.id,
          metadata={"new_status": app.status})
    return jsonify({"id": app.id, "status": "updated"})


@tracker_bp.route("/api/tracker/<app_id>", methods=["DELETE"])
@login_required
def delete_application(app_id):
    app = JobApplication.query.filter_by(id=app_id, user_id=current_user.id).first()
    if not app:
        return jsonify({"error": "Application not found"}), 404

    db.session.delete(app)
    db.session.commit()
    track("tracker.application.deleted", category="feature", user_id=current_user.id)
    return jsonify({"status": "deleted"})
