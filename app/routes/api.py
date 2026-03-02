"""Core API routes — tailoring, progress, results, downloads."""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import uuid
from pathlib import Path

import pdfplumber
from flask import Blueprint, Response, jsonify, request
from flask_login import current_user, login_required

from app.extensions import csrf, db, limiter
from app.models import TailoringJob
from app.services.admin_config import AdminConfigManager
from app.services.file_service import serve_download
from app.services.pipeline import (
    MAX_QUEUE_DEPTH,
    cleanup_old_jobs,
    jobs,
    jobs_lock,
    pipeline_queue_depth,
    pipeline_queue_lock,
    run_pipeline_job,
)
from app.services.usage import usage_tracker
from config import DEFAULT_MODEL
from utils import create_output_dir

logger = logging.getLogger("cvtailro.api")

api_bp = Blueprint("api", __name__)

# Exempt multipart upload and SSE from CSRF (they use custom auth)
csrf.exempt(api_bp)


@api_bp.route("/api/tailor", methods=["POST"])
@limiter.limit("10 per hour")
def start_tailoring():
    from flask import current_app

    cleanup_old_jobs()

    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()
    if not api_key:
        return jsonify({"error": "Service not configured. An admin must set the API key at /admin."}), 400

    if admin_config.allow_user_model_selection:
        model = request.form.get("model", admin_config.default_model or DEFAULT_MODEL).strip()
    else:
        model = admin_config.default_model or DEFAULT_MODEL

    with pipeline_queue_lock:
        if pipeline_queue_depth >= MAX_QUEUE_DEPTH:
            return jsonify({"error": "Server is at capacity. Please try again in a few minutes."}), 503

    client_ip = request.remote_addr or "unknown"
    rate_key = f"user:{current_user.id}" if current_user.is_authenticated else f"ip:{client_ip}"
    if not usage_tracker.check_and_record(rate_key, admin_config.rate_limit_per_hour):
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    resume_file = request.files["resume"]
    job_text = request.form.get("job_description", "").strip()
    mode = request.form.get("mode", "conservative")
    template = request.form.get("template", "modern")

    if not resume_file.filename:
        return jsonify({"error": "No resume file selected"}), 400

    job_text = re.sub(r"<[^>]+>", "", job_text)
    if len(job_text) > 50000:
        return jsonify({"error": "Job description is too long (maximum 50,000 characters)"}), 400
    if not job_text or len(job_text) < 50:
        return jsonify({"error": "Job description is too short (minimum 50 characters)"}), 400
    if mode not in ("conservative", "aggressive"):
        return jsonify({"error": "Invalid mode"}), 400
    if template not in ("executive", "modern", "minimal"):
        return jsonify({"error": "Invalid template"}), 400

    resume_ext = Path(resume_file.filename).suffix.lower()
    if resume_ext == ".pdf":
        resume_file.stream.seek(0)
        magic_bytes = resume_file.stream.read(5)
        resume_file.stream.seek(0)
        if magic_bytes != b"%PDF-":
            return jsonify({"error": "File does not appear to be a valid PDF (bad magic bytes)"}), 400
        try:
            resume_file.stream.seek(0)
            with pdfplumber.open(resume_file.stream) as pdf:
                if not pdf.pages:
                    return jsonify({"error": "PDF is empty (no pages)"}), 400
                test_text = pdf.pages[0].extract_text()
                if not test_text or len(test_text.strip()) < 20:
                    return (
                        jsonify({"error": "PDF appears to be image-based or empty. Please use a text-based PDF."}),
                        400,
                    )
            resume_file.stream.seek(0)
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {e}"}), 400
    elif resume_ext not in (".md", ".txt"):
        return jsonify({"error": "Unsupported file type. Use PDF, MD, or TXT."}), 400

    job_id = uuid.uuid4().hex[:16]
    output_dir = create_output_dir(job_id=job_id)

    resume_path = output_dir / f"input_resume{resume_ext}"
    resume_file.save(str(resume_path))
    (output_dir / "input_job_description.txt").write_text(job_text, encoding="utf-8")

    user_id = current_user.id if current_user.is_authenticated else None

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "queue": queue.Queue(),
            "output_dir": str(output_dir),
            "created_at": __import__("time").time(),
            "user_id": user_id,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=run_pipeline_job,
        args=(
            current_app._get_current_object(),
            job_id,
            str(resume_path),
            job_text,
            mode,
            template,
            output_dir,
            api_key,
            model,
            user_id,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@api_bp.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    with jobs_lock:
        job_data = jobs.get(job_id)
        if job_data is not None:
            job_user_id = job_data.get("user_id")
            if job_user_id is not None:
                if not current_user.is_authenticated:
                    return jsonify({"error": "Job not found"}), 404
                if job_user_id != current_user.id:
                    return jsonify({"error": "Job not found"}), 404
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404

    def generate():
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'status': 'error', 'detail': 'Job has been cleaned up'})}\n\n"
                return
            q = job["queue"]
        while True:
            try:
                event = q.get(timeout=10)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("complete", "error"):
                    break
            except queue.Empty:
                with jobs_lock:
                    if job_id not in jobs:
                        yield f"data: {json.dumps({'status': 'error', 'detail': 'Job expired'})}\n\n"
                        return
                yield f"data: {json.dumps({'status': 'keepalive'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_bp.route("/api/result/<job_id>")
def get_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is not None and current_user.is_authenticated:
            if job.get("user_id") and job["user_id"] != current_user.id:
                return jsonify({"error": "Job not found"}), 404

    if job is not None:
        if job["status"] == "running":
            return jsonify({"status": "running"}), 202
        if job["status"] == "error":
            return jsonify({"status": "error", "error": job["error"]}), 500
        if job.get("result") is not None:
            return jsonify({"status": "complete", "result": job["result"]})

    if current_user.is_authenticated:
        db_job = TailoringJob.query.filter_by(id=job_id, user_id=current_user.id).first()
    else:
        db_job = db.session.get(TailoringJob, job_id)
        if db_job is not None and db_job.user_id is not None:
            db_job = None  # Anonymous users cannot access authenticated users' jobs
    if db_job is None:
        return jsonify({"error": "Job not found"}), 404

    if db_job.status == "running":
        return jsonify({"status": "running"}), 202
    if db_job.status == "error":
        return jsonify({"status": "error", "error": db_job.error_message or "Unknown error"}), 500

    result = {
        "match_score": db_job.match_score,
        "cosine_similarity": db_job.cosine_similarity,
        "missing_keywords": db_job.missing_keywords,
        "rewrite_mode": db_job.rewrite_mode,
        "template": db_job.template,
        "job_title": db_job.job_title,
        "company": db_job.company,
        "ats_resume_md": db_job.ats_resume_md,
        "recruiter_resume_md": db_job.recruiter_resume_md,
        "talking_points_md": db_job.talking_points_md,
        "cover_letter_md": db_job.cover_letter_md,
        "section_scores": db_job.section_scores,
        "resume_quality": db_job.resume_quality_json,
        "email_templates_md": db_job.email_templates_md,
        "keyword_density": db_job.keyword_density_json,
        "files": [f.filename for f in db_job.files],
    }
    if db_job.original_match_score is not None:
        result["original_match_score"] = db_job.original_match_score
    if db_job.match_score is not None:
        result["tailored_match_score"] = db_job.match_score
    return jsonify({"status": "complete", "result": result})


@api_bp.route("/api/download/<job_id>/<filename>")
def download_file(job_id: str, filename: str):
    return serve_download(job_id, filename)


@api_bp.route("/api/download-check/<job_id>")
@login_required
def download_check(job_id: str):
    from storage import r2_storage as r2

    result = {
        "job_id": job_id,
        "user_id": current_user.id,
        "in_memory": False,
        "local_files": [],
        "r2_configured": r2.is_configured,
        "r2_files": [],
        "db_found": False,
        "db_status": None,
        "db_has_ats_md": False,
        "db_has_rec_md": False,
        "db_has_tp_md": False,
        "db_user_id": None,
    }

    with jobs_lock:
        if job_id in jobs:
            result["in_memory"] = True
            output_dir = jobs[job_id]["output_dir"]
            try:
                result["local_files"] = [f.name for f in Path(output_dir).iterdir() if f.is_file()]
            except Exception:
                pass

    if r2.is_configured:
        from app.models import JobFile

        job_files = JobFile.query.filter_by(job_id=job_id).all()
        result["r2_files"] = [jf.filename for jf in job_files]

    db_job = db.session.get(TailoringJob, job_id)
    if db_job:
        result["db_found"] = True
        result["db_status"] = db_job.status
        result["db_has_ats_md"] = bool(db_job.ats_resume_md)
        result["db_has_rec_md"] = bool(db_job.recruiter_resume_md)
        result["db_has_tp_md"] = bool(db_job.talking_points_md)
        result["db_user_id"] = db_job.user_id
        result["db_user_match"] = db_job.user_id == current_user.id

    return jsonify(result)
