"""Saved resumes API — CRUD for user's reusable master resumes."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from app.extensions import db
from app.models import SavedResume
from app.services.telemetry import track

saved_resumes_bp = Blueprint("saved_resumes", __name__)


@saved_resumes_bp.route("/api/saved-resumes", methods=["GET"])
@login_required
def list_saved_resumes():
    resumes = (
        SavedResume.query.filter_by(user_id=current_user.id).order_by(SavedResume.updated_at.desc()).all()
    )
    return jsonify(
        {
            "resumes": [
                {
                    "id": r.id,
                    "name": r.name,
                    "preview": r.resume_text[:200] if r.resume_text else "",
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in resumes
            ],
        }
    )


@saved_resumes_bp.route("/api/saved-resumes", methods=["POST"])
@login_required
def save_resume():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "").strip()
    name = data.get("name", "My Resume").strip()[:255]
    resume_id = data.get("id")

    if not resume_text:
        return jsonify({"error": "Resume text is required"}), 400

    if resume_id:
        resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first()
        if not resume:
            return jsonify({"error": "Resume not found"}), 404
        resume.resume_text = resume_text
        resume.name = name
        resume.updated_at = datetime.now(timezone.utc)
    else:
        count = SavedResume.query.filter_by(user_id=current_user.id).count()
        if count >= 10:
            return (
                jsonify({"error": "Maximum 10 saved resumes. Delete one to save a new one."}),
                400,
            )
        resume = SavedResume(user_id=current_user.id, name=name, resume_text=resume_text)
        db.session.add(resume)

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Failed to save resume"}), 500
    action = "saved_resume.updated" if resume_id else "saved_resume.created"
    track(action, category="feature", user_id=current_user.id)
    return jsonify({"ok": True, "id": resume.id})


@saved_resumes_bp.route("/api/saved-resumes/<resume_id>", methods=["GET"])
@login_required
def get_saved_resume(resume_id: str):
    resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    return jsonify(
        {
            "id": resume.id,
            "name": resume.name,
            "resume_text": resume.resume_text,
        }
    )


@saved_resumes_bp.route("/api/saved-resumes/<resume_id>", methods=["DELETE"])
@login_required
def delete_saved_resume(resume_id: str):
    resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(resume)
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({"error": "Failed to delete resume"}), 500
    track("saved_resume.deleted", category="feature", user_id=current_user.id)
    return jsonify({"ok": True})
