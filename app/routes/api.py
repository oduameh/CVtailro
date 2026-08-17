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
    cleanup_old_jobs,
    jobs,
    jobs_lock,
    queue_is_full,
    run_pipeline_job,
)
from app.services.telemetry import track
from app.services.usage import daily_jobs_used, usage_tracker
from config import DEFAULT_MODEL, RECOMMENDED_MODELS
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


def _resolve_model(admin_config) -> str:
    """Resolve the model for a job, honouring user selection only when allowed.

    User-supplied values are validated against the curated catalog — the admin
    pays all API costs, so an arbitrary form value must never reach OpenRouter
    (e.g. a hand-crafted request billing a frontier model to the admin's key).
    Unknown values silently fall back to the admin default, which itself may be
    a custom ID the admin chose deliberately.
    """
    default = admin_config.default_model or DEFAULT_MODEL
    if not admin_config.allow_user_model_selection:
        return default
    requested = (request.form.get("model") or "").strip()
    if requested and requested in set(RECOMMENDED_MODELS.values()):
        return requested
    return default


def _daily_budget_response(admin_config, rate_key: str, needed: int = 1):
    """Return a 429 JSON response if over the per-user daily job cap, else None.

    Admins are exempt. Authenticated users are counted durably from the
    TailoringJob table; anonymous users via an IP-keyed rolling 24h counter.
    The cap (``daily_job_limit``, 0 = unlimited) is admin-configurable.
    """
    if current_user.is_authenticated and getattr(current_user, "is_admin", False):
        return None
    try:
        limit = int(getattr(admin_config, "daily_job_limit", 0) or 0)
    except (TypeError, ValueError):
        limit = 0  # defensive: unparseable/mocked config → treat as unlimited
    if limit <= 0:
        return None

    if current_user.is_authenticated:
        # Authenticated users are counted from the durable TailoringJob table.
        # The row is inserted by the background pipeline, so this is a soft cap
        # with a small check→insert window; it self-corrects on the next request
        # and the burst is bounded by the hourly limiter.
        used = daily_jobs_used(current_user.id)
        over_limit = used + needed > limit
    else:
        # Anonymous users: atomic check-and-record (no check→record gap in-memory).
        used = usage_tracker.daily_count(rate_key)
        over_limit = not usage_tracker.check_and_record_daily(rate_key, limit, needed)

    if over_limit:
        uid = current_user.id if current_user.is_authenticated else None
        track(
            "tailor.request.rejected",
            category="tailor",
            user_id=uid,
            metadata={"reason": "daily_budget", "used_today": used, "daily_limit": limit},
        )
        noun = "resume" if limit == 1 else "resumes"
        msg = (
            f"You've reached your daily limit of {limit} tailored {noun}. "
            "The limit resets at midnight UTC — come back tomorrow!"
        )
        return jsonify({"error": msg, "daily_limit": limit, "used_today": used}), 429
    return None


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

    model = _resolve_model(admin_config)

    if queue_is_full():
        track("tailor.request.rejected", category="tailor", user_id=uid, metadata={"reason": "queue_full"})
        return jsonify({"error": "Server is at capacity. Please try again in a few minutes."}), 503

    client_ip = request.remote_addr or "unknown"
    rate_key = f"user:{current_user.id}" if current_user.is_authenticated else f"ip:{client_ip}"
    if not usage_tracker.check_and_record(rate_key, admin_config.rate_limit_per_hour):
        track("tailor.request.rejected", category="tailor", user_id=uid, metadata={"reason": "rate_limited"})
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    # Resume source: an uploaded file OR a previously saved resume (saved_resume_id).
    saved_resume_id = (request.form.get("saved_resume_id") or "").strip()
    resume_file = request.files.get("resume")
    saved = None
    if saved_resume_id:
        if not current_user.is_authenticated:
            return jsonify({"error": "Sign in to use saved resumes"}), 401
        from app.models import SavedResume

        saved = SavedResume.query.filter_by(id=saved_resume_id, user_id=current_user.id).first()
        if saved is None or not (saved.resume_text or "").strip():
            return jsonify({"error": "Saved resume not found"}), 404
    elif resume_file is None or not resume_file.filename:
        return jsonify({"error": "No resume file uploaded"}), 400

    job_text = request.form.get("job_description", "").strip()
    mode = request.form.get("mode", "conservative")
    template = request.form.get("template", "modern")

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

    if saved_resume_id:
        resume_ext = ".txt"
    else:
        error, resume_ext = _validate_resume_file(resume_file)
        if error:
            return jsonify({"error": error}), 400

    budget_resp = _daily_budget_response(admin_config, rate_key, needed=1)
    if budget_resp:
        return budget_resp

    # Anonymous daily usage was already recorded atomically in the budget check.
    job_id = uuid.uuid4().hex[:16]
    output_dir = create_output_dir(job_id=job_id)

    resume_path = output_dir / f"input_resume{resume_ext}"
    if saved_resume_id:
        resume_path.write_text(saved.resume_text, encoding="utf-8")
    else:
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

    track(
        "tailor.job.created",
        category="tailor",
        user_id=uid,
        job_id=job_id,
        metadata={
            "model": model,
            "mode": mode,
            "template": template,
            "resume_ext": resume_ext,
            "resume_source": "saved" if saved_resume_id else "upload",
        },
    )
    return jsonify({"job_id": job_id})


# Concurrent SSE listeners per job. Each open stream pins a Gunicorn request
# thread for the job's lifetime, so without a cap one anonymous job + a handful
# of EventSource connections can exhaust the whole thread pool (including
# /api/health, which Railway's healthcheck depends on).
MAX_STREAMS_PER_JOB = 3
MAX_KEEPALIVE_CYCLES = 90  # 90 x 10s queue timeout = 15 min absolute stream cap
_stream_counts: dict[str, int] = {}


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
        if _stream_counts.get(job_id, 0) >= MAX_STREAMS_PER_JOB:
            return jsonify({"error": "Too many progress connections for this job"}), 429

    def _release_stream_slot():
        with jobs_lock:
            remaining = _stream_counts.get(job_id, 1) - 1
            if remaining <= 0:
                _stream_counts.pop(job_id, None)
            else:
                _stream_counts[job_id] = remaining

    def generate():
        # Increment inside the generator (not the route) so the paired
        # decrement in `finally` is guaranteed: a generator that is never
        # iterated runs neither, one that starts runs both.
        with jobs_lock:
            _stream_counts[job_id] = _stream_counts.get(job_id, 0) + 1
        try:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is None:
                    yield f"data: {json.dumps({'status': 'error', 'detail': 'Job has been cleaned up'})}\n\n"
                    return
                # A (re)connecting client must not block on the queue when the
                # job already finished — the single terminal event is consumed
                # destructively, so a second reader would hang on keepalives
                # until the job entry expires.
                status = job.get("status")
                if status == "complete":
                    yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                    return
                if status == "error":
                    detail = job.get("error") or "Pipeline failed"
                    yield f"data: {json.dumps({'status': 'error', 'detail': detail})}\n\n"
                    return
                q = job["queue"]
            idle_cycles = 0
            while True:
                try:
                    event = q.get(timeout=10)
                    idle_cycles = 0
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("complete", "error"):
                        break
                except queue.Empty:
                    idle_cycles += 1
                    with jobs_lock:
                        if job_id not in jobs:
                            yield f"data: {json.dumps({'status': 'error', 'detail': 'Job expired'})}\n\n"
                            return
                    if idle_cycles >= MAX_KEEPALIVE_CYCLES:
                        # Absolute cap so a stream can never pin a request
                        # thread indefinitely; the client falls back to
                        # reconnecting or polling /api/result.
                        yield f"data: {json.dumps({'status': 'error', 'detail': 'Progress stream timed out'})}\n\n"
                        return
                    yield f"data: {json.dumps({'status': 'keepalive'})}\n\n"
        finally:
            _release_stream_slot()

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
        "recruiter_resume_md": db_job.recruiter_resume_md or db_job.ats_resume_md,
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
        logger.warning(f"Could not read uploaded file: {e}")
        return jsonify({"error": "Could not read uploaded file"}), 400

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
@login_required
@limiter.limit("30 per hour")
def boost_bullet():
    """Rewrite a single bullet point with stronger action verbs and metrics.

    Requires login: each call spends admin-paid OpenRouter credits, and the
    per-IP rate limit alone is trivially bypassed by rotating IPs.
    """
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

    cleanup_old_jobs()

    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()
    uid = current_user.id if current_user.is_authenticated else None
    if not api_key:
        return jsonify({"error": "Service not configured"}), 400

    model = _resolve_model(admin_config)

    if queue_is_full():
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
    if not resume_file.filename:
        return jsonify({"error": "No resume file selected"}), 400

    job_descriptions_raw = request.form.get("job_descriptions", "").strip()
    if not job_descriptions_raw:
        return jsonify({"error": "No job descriptions provided"}), 400

    # Parse job descriptions separated by "---" (same sanitisation as /api/tailor)
    job_descriptions = []
    for jd in job_descriptions_raw.split("---"):
        jd = re.sub(r"<[^>]+>", "", jd).strip()
        if not jd or len(jd) < 50:
            continue
        if len(jd) > 50000:
            return jsonify({"error": "A job description is too long (maximum 50,000 characters)"}), 400
        job_descriptions.append(jd)
    if len(job_descriptions) < 2:
        return jsonify({"error": "Provide at least 2 job descriptions separated by ---"}), 400
    if len(job_descriptions) > 5:
        return jsonify({"error": "Maximum 5 job descriptions per batch"}), 400

    mode = request.form.get("mode", "conservative")
    template = request.form.get("template", "modern")
    if mode not in ("conservative", "aggressive"):
        return jsonify({"error": "Invalid mode"}), 400
    from pdf_generator import ALL_TEMPLATE_NAMES

    if template not in ALL_TEMPLATE_NAMES:
        return jsonify({"error": "Invalid template"}), 400

    error, resume_ext = _validate_resume_file(resume_file)
    if error:
        return jsonify({"error": error}), 400

    budget_resp = _daily_budget_response(admin_config, rate_key, needed=len(job_descriptions))
    if budget_resp:
        return budget_resp

    # Anonymous daily usage for all batch jobs was recorded atomically in the
    # budget check (needed=len(job_descriptions)).
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

        track(
            "tailor.job.created",
            category="tailor",
            user_id=uid,
            job_id=job_id,
            metadata={
                "model": model,
                "mode": mode,
                "template": template,
                "resume_ext": resume_ext,
                "resume_source": "upload",
                "batch": True,
                "batch_index": i,
            },
        )

    return jsonify({"job_ids": job_ids, "count": len(job_ids)})


@api_bp.route("/api/download/<job_id>/<filename>")
@limiter.limit("30 per minute")
def download_file(job_id: str, filename: str):
    uid = current_user.id if current_user.is_authenticated else None
    ext = Path(filename).suffix.lower()
    track(
        "download.requested",
        category="download",
        user_id=uid,
        job_id=job_id,
        metadata={"filename_ext": ext, "filename": filename},
    )
    return serve_download(job_id, filename)


@api_bp.route("/api/download-check/<job_id>")
@login_required
def download_check(job_id: str):
    from storage import r2_storage as r2

    # Ownership gate: file listings embed job metadata (smart filenames carry
    # role + employer), so only the job's owner — or an admin — may see them.
    # Anonymous jobs (user_id None) stay visible, matching download semantics.
    is_admin = getattr(current_user, "is_admin", False)
    with jobs_lock:
        mem_job = jobs.get(job_id)
        mem_owner = mem_job.get("user_id") if mem_job else None
    owns_mem = mem_job is not None and (mem_owner is None or mem_owner == current_user.id)
    db_job = TailoringJob.query.filter_by(id=job_id, user_id=current_user.id).first()
    if not is_admin and not owns_mem and db_job is None:
        return jsonify({"error": "Job not found"}), 404

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

    if mem_job is not None and (owns_mem or is_admin):
        result["in_memory"] = True
        try:
            result["local_files"] = [f.name for f in Path(mem_job["output_dir"]).iterdir() if f.is_file()]
        except Exception:
            pass

    if r2.is_configured:
        from app.models import JobFile

        job_files = JobFile.query.filter_by(job_id=job_id).all()
        result["r2_files"] = [jf.filename for jf in job_files]

    if db_job:
        result["db_found"] = True
        result["db_status"] = db_job.status
        result["db_has_ats_md"] = bool(db_job.ats_resume_md)
        result["db_has_rec_md"] = bool(db_job.recruiter_resume_md)
        result["db_has_tp_md"] = bool(db_job.talking_points_md)

    return jsonify(result)
