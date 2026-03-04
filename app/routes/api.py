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

from app.extensions import db, limiter
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
from app.services.telemetry import track
from app.services.usage import usage_tracker
from config import DEFAULT_MODEL
from utils import create_output_dir

logger = logging.getLogger("cvtailro.api")

api_bp = Blueprint("api", __name__)


def _validate_resume_file(resume_file) -> tuple[str | None, str]:
    """Validate resume file. Returns (error_message, extension). error_message is None if valid."""
    ext = Path(resume_file.filename).suffix.lower()
    if ext == ".pdf":
        resume_file.stream.seek(0)
        magic_bytes = resume_file.stream.read(5)
        resume_file.stream.seek(0)
        if magic_bytes != b"%PDF-":
            return "File does not appear to be a valid PDF (bad magic bytes)", ext
        try:
            resume_file.stream.seek(0)
            with pdfplumber.open(resume_file.stream) as pdf:
                if not pdf.pages:
                    return "PDF is empty (no pages)", ext
                test_text = pdf.pages[0].extract_text()
                if not test_text or len(test_text.strip()) < 20:
                    return "PDF appears to be image-based or empty. Please use a text-based PDF.", ext
            resume_file.stream.seek(0)
        except Exception:
            return "Could not read PDF. Please ensure the file is a valid text-based PDF.", ext
    elif ext not in (".md", ".txt"):
        return "Unsupported file type. Use PDF, MD, or TXT.", ext
    return None, ext


@api_bp.route("/api/tailor", methods=["POST"])
@limiter.limit("10 per hour")
def start_tailoring():
    from flask import current_app

    cleanup_old_jobs()

    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()
    uid = current_user.id if current_user.is_authenticated else None

    if not api_key:
        track("tailor.request.rejected", category="tailor", user_id=uid, metadata={"reason": "no_api_key"})
        return jsonify({"error": "Service not configured. An admin must set the API key at /admin."}), 400

    if admin_config.allow_user_model_selection:
        model = request.form.get("model", admin_config.default_model or DEFAULT_MODEL).strip()
    else:
        model = admin_config.default_model or DEFAULT_MODEL

    with pipeline_queue_lock:
        if pipeline_queue_depth >= MAX_QUEUE_DEPTH:
            track("tailor.request.rejected", category="tailor", user_id=uid, metadata={"reason": "queue_full"})
            return jsonify({"error": "Server is at capacity. Please try again in a few minutes."}), 503

    client_ip = request.remote_addr or "unknown"
    rate_key = f"user:{current_user.id}" if current_user.is_authenticated else f"ip:{client_ip}"
    if not usage_tracker.check_and_record(rate_key, admin_config.rate_limit_per_hour):
        track("tailor.request.rejected", category="tailor", user_id=uid, metadata={"reason": "rate_limited"})
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
    from pdf_generator import ALL_TEMPLATE_NAMES

    if template not in ALL_TEMPLATE_NAMES:
        return jsonify({"error": "Invalid template"}), 400

    error, resume_ext = _validate_resume_file(resume_file)
    if error:
        return jsonify({"error": error}), 400

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

    track("tailor.job.created", category="tailor", user_id=uid, job_id=job_id,
          metadata={"model": model, "mode": mode, "template": template, "resume_ext": resume_ext})
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
        if job is not None:
            job_user_id = job.get("user_id")
            if job_user_id is not None:
                if not current_user.is_authenticated:
                    return jsonify({"error": "Job not found"}), 404
                if job_user_id != current_user.id:
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
        "recruiter_resume_md": db_job.ats_resume_md,  # deprecated: same as ats_resume_md
        "original_resume_text": db_job.original_resume_text,
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


@api_bp.route("/api/score-resume", methods=["POST"])
@limiter.limit("30 per hour")
def score_resume():
    """Score a resume without tailoring — standalone quality check."""
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    resume_file = request.files["resume"]
    if not resume_file.filename:
        return jsonify({"error": "No resume file selected"}), 400

    resume_ext = Path(resume_file.filename).suffix.lower()
    if resume_ext not in (".pdf", ".md", ".txt"):
        return jsonify({"error": "Unsupported file type. Use PDF, MD, or TXT."}), 400

    try:
        if resume_ext == ".pdf":
            resume_file.stream.seek(0)
            with pdfplumber.open(resume_file.stream) as pdf:
                resume_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            resume_file.stream.seek(0)
            resume_text = resume_file.stream.read().decode("utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not resume_text or len(resume_text.strip()) < 50:
        return jsonify({"error": "Resume appears to be empty or too short"}), 400

    from resume_quality import analyze_resume, extract_bullets_from_markdown

    bullets = extract_bullets_from_markdown(resume_text)
    if not bullets:
        return jsonify({"error": "No bullet points found in resume"}), 400

    report = analyze_resume(bullets)

    return jsonify(
        {
            "overall_score": report.overall_score,
            "total_bullets": report.total_bullets,
            "bullets_with_metrics": report.bullets_with_metrics,
            "metrics_percentage": report.metrics_percentage,
            "unique_verbs": report.unique_verbs,
            "repeated_verbs": report.repeated_verbs,
            "weak_verbs_used": report.weak_verbs_used,
            "filler_words_found": report.filler_words_found,
            "avg_bullet_length": report.avg_bullet_length,
            "too_long_bullets": report.too_long_bullets,
            "too_short_bullets": report.too_short_bullets,
            "improvement_summary": report.improvement_summary,
            "bullet_analyses": [
                {
                    "text": ba.text,
                    "score": ba.score,
                    "has_metrics": ba.has_metrics,
                    "verb_strength": ba.verb_strength,
                    "action_verb": ba.action_verb,
                    "suggestions": ba.suggestions,
                }
                for ba in report.bullet_analyses[:50]
            ],
        }
    )


@api_bp.route("/api/boost-bullet", methods=["POST"])
@limiter.limit("30 per hour")
def boost_bullet():
    """Rewrite a single bullet point with stronger action verbs and metrics."""
    data = request.get_json(silent=True)
    if not data or not data.get("bullet"):
        return jsonify({"error": "No bullet text provided"}), 400

    bullet = data["bullet"].strip()
    if len(bullet) < 10:
        return jsonify({"error": "Bullet text too short"}), 400
    if len(bullet) > 500:
        return jsonify({"error": "Bullet text too long (max 500 chars)"}), 400

    job_title = data.get("job_title", "").strip()

    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()
    if not api_key:
        return jsonify({"error": "Service not configured"}), 400

    model = admin_config.default_model or DEFAULT_MODEL

    context = f" for a {job_title} role" if job_title else ""
    prompt = (
        f"Rewrite this resume bullet point to be more impactful{context}. "
        "Use a strong action verb, include quantifiable metrics where possible, "
        "and keep it concise (under 30 words). Do NOT fabricate specific numbers "
        "that weren't implied. Return ONLY the rewritten bullet, nothing else.\n\n"
        f"Original: {bullet}"
    )

    try:
        import requests as http_requests

        resp = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert resume writer. Rewrite bullet points to be more impactful.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return jsonify({"error": "LLM API call failed"}), 500
        improved = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("- ")
        if not improved:
            return jsonify({"error": "Failed to generate improved bullet"}), 500

        from resume_quality import analyze_bullet

        original_analysis = analyze_bullet(bullet)
        improved_analysis = analyze_bullet(improved)

        return jsonify(
            {
                "original": bullet,
                "improved": improved,
                "original_score": original_analysis.score,
                "improved_score": improved_analysis.score,
                "original_has_metrics": original_analysis.has_metrics,
                "improved_has_metrics": improved_analysis.has_metrics,
                "suggestions": improved_analysis.suggestions,
            }
        )
    except Exception:
        logger.exception("Bullet boost failed")
        return jsonify({"error": "Failed to improve bullet"}), 500


@api_bp.route("/api/batch-tailor", methods=["POST"])
@limiter.limit("3 per hour")
def start_batch_tailoring():
    """Start tailoring a resume against multiple job descriptions."""
    from flask import current_app

    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()
    if not api_key:
        return jsonify({"error": "Service not configured"}), 400

    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    resume_file = request.files["resume"]
    if not resume_file.filename:
        return jsonify({"error": "No resume file selected"}), 400

    job_descriptions_raw = request.form.get("job_descriptions", "").strip()
    if not job_descriptions_raw:
        return jsonify({"error": "No job descriptions provided"}), 400

    # Parse job descriptions separated by "---"
    job_descriptions = [
        jd.strip() for jd in job_descriptions_raw.split("---") if jd.strip() and len(jd.strip()) >= 50
    ]
    if len(job_descriptions) < 2:
        return jsonify({"error": "Provide at least 2 job descriptions separated by ---"}), 400
    if len(job_descriptions) > 5:
        return jsonify({"error": "Maximum 5 job descriptions per batch"}), 400

    mode = request.form.get("mode", "conservative")
    template = request.form.get("template", "modern")

    error, resume_ext = _validate_resume_file(resume_file)
    if error:
        return jsonify({"error": error}), 400

    job_ids = []
    for i, jd in enumerate(job_descriptions):
        job_id = uuid.uuid4().hex[:16]
        output_dir = create_output_dir(job_id=job_id)

        resume_file.stream.seek(0)
        resume_path = output_dir / f"input_resume{resume_ext}"
        resume_file.save(str(resume_path))
        (output_dir / "input_job_description.txt").write_text(jd, encoding="utf-8")

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
                "batch_index": i,
            }

        model = admin_config.default_model or DEFAULT_MODEL
        thread = threading.Thread(
            target=run_pipeline_job,
            args=(
                current_app._get_current_object(),
                job_id,
                str(resume_path),
                jd,
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
        job_ids.append(job_id)

    return jsonify({"job_ids": job_ids, "count": len(job_ids)})


@api_bp.route("/api/download/<job_id>/<filename>")
@limiter.limit("30 per minute")
def download_file(job_id: str, filename: str):
    uid = current_user.id if current_user.is_authenticated else None
    ext = Path(filename).suffix.lower()
    track("download.requested", category="download", user_id=uid, job_id=job_id,
          metadata={"filename_ext": ext, "filename": filename})
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

    db_job = TailoringJob.query.filter_by(id=job_id, user_id=current_user.id).first()
    if db_job:
        result["db_found"] = True
        result["db_status"] = db_job.status
        result["db_has_ats_md"] = bool(db_job.ats_resume_md)
        result["db_has_rec_md"] = bool(db_job.recruiter_resume_md)
        result["db_has_tp_md"] = bool(db_job.talking_points_md)

    return jsonify(result)
