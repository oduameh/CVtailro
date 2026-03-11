"""History API routes — paginated job history for authenticated users."""

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from app.models import TailoringJob

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history")
@login_required
def get_history():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 50)
    query = TailoringJob.query.filter_by(user_id=current_user.id).order_by(TailoringJob.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify(
        {
            "jobs": [
                {
                    "id": j.id,
                    "status": j.status,
                    "job_title": j.job_title,
                    "company": j.company,
                    "match_score": j.match_score,
                    "rewrite_mode": j.rewrite_mode,
                    "template": j.template,
                    "model_used": j.model_used,
                    "job_description_snippet": j.job_description_snippet,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "has_resume": bool(j.ats_resume_md),
                    "files": [{"filename": f.filename, "size_bytes": f.size_bytes} for f in j.files],
                }
                for j in pagination.items
            ],
            "total": pagination.total,
            "page": pagination.page,
            "pages": pagination.pages,
        }
    )


@history_bp.route("/api/history/<job_id>")
@login_required
def get_history_job(job_id: str):
    job = TailoringJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()
    return jsonify(
        {
            "id": job.id,
            "status": job.status,
            "job_title": job.job_title,
            "company": job.company,
            "match_score": job.match_score,
            "original_match_score": job.original_match_score,
            "tailored_match_score": job.match_score,
            "cosine_similarity": job.cosine_similarity,
            "missing_keywords": job.missing_keywords,
            "rewrite_mode": job.rewrite_mode,
            "template": job.template,
            "model_used": job.model_used,
            "job_description_snippet": job.job_description_snippet,
            "job_description_full": job.job_description_full,
            "original_resume_text": job.original_resume_text,
            "ats_resume_md": job.ats_resume_md,
            "recruiter_resume_md": job.recruiter_resume_md or job.ats_resume_md,
            "talking_points_md": job.talking_points_md,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "duration_seconds": job.duration_seconds,
            "files": [{"filename": f.filename, "size_bytes": f.size_bytes} for f in job.files],
        }
    )
