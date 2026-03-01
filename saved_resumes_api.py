"""Saved Resumes API — lets users save and reuse their master resumes."""

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from database import db, SavedResume

saved_resumes_bp = Blueprint("saved_resumes", __name__)


@saved_resumes_bp.route("/api/saved-resumes", methods=["GET"])
@login_required
def list_saved_resumes():
    """List user's saved resumes."""
    resumes = SavedResume.query.filter_by(user_id=current_user.id).order_by(
        SavedResume.updated_at.desc()
    ).all()
    return jsonify({
        "resumes": [{
            "id": r.id,
            "name": r.name,
            "preview": r.resume_text[:200] if r.resume_text else "",
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        } for r in resumes],
    })


@saved_resumes_bp.route("/api/saved-resumes", methods=["POST"])
@login_required
def save_resume():
    """Save a new resume or update existing."""
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "").strip()
    name = data.get("name", "My Resume").strip()[:255]
    resume_id = data.get("id")

    if not resume_text:
        return jsonify({"error": "Resume text is required"}), 400

    if resume_id:
        # Update existing
        resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first()
        if not resume:
            return jsonify({"error": "Resume not found"}), 404
        resume.resume_text = resume_text
        resume.name = name
        from datetime import datetime, timezone
        resume.updated_at = datetime.now(timezone.utc)
    else:
        # Create new (limit to 10 per user)
        count = SavedResume.query.filter_by(user_id=current_user.id).count()
        if count >= 10:
            return jsonify({"error": "Maximum 10 saved resumes. Delete one to save a new one."}), 400
        resume = SavedResume(user_id=current_user.id, name=name, resume_text=resume_text)
        db.session.add(resume)

    db.session.commit()
    return jsonify({"ok": True, "id": resume.id})


@saved_resumes_bp.route("/api/saved-resumes/<resume_id>", methods=["GET"])
@login_required
def get_saved_resume(resume_id):
    """Get full text of a saved resume."""
    resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    return jsonify({
        "id": resume.id,
        "name": resume.name,
        "resume_text": resume.resume_text,
    })


@saved_resumes_bp.route("/api/saved-resumes/<resume_id>", methods=["DELETE"])
@login_required
def delete_saved_resume(resume_id):
    """Delete a saved resume."""
    resume = SavedResume.query.filter_by(id=resume_id, user_id=current_user.id).first_or_404()
    db.session.delete(resume)
    db.session.commit()
    return jsonify({"ok": True})
