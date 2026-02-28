#!/usr/bin/env python3
"""
CVtailro Web UI

A local Flask application providing a browser-based interface to the
multi-agent resume tailoring pipeline. Upload a PDF resume, paste a
job description, and get tailored resumes with match reports.

Usage:
    python app.py
    # Then open http://localhost:5050
"""

from __future__ import annotations

import gc
import json
import logging
import mimetypes
import os
import queue
import re
import resource
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
import requests as http_requests
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
)

from flask_login import current_user, login_required
from flask_migrate import Migrate

from admin_config import AdminConfigManager
from analytics import pipeline_analytics
from agents.ats_optimiser import ATSOptimiserAgent
from agents.bullet_optimiser import BulletOptimiserAgent
from agents.final_assembly import FinalAssemblyAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.job_intelligence import JobIntelligenceAgent
from agents.recruiter_optimiser import RecruiterOptimiserAgent
from agents.resume_parser import ResumeParserAgent
from config import AppConfig, DEFAULT_MODEL, RECOMMENDED_MODELS
from models import RewriteMode
from database import db, TailoringJob, JobFile
from auth import auth_bp, login_manager, init_oauth
from storage import r2_storage
from pdf_generator import generate_resume_pdf
from docx_generator import generate_resume_docx
from similarity import resume_job_similarity
from utils import (
    create_output_dir,
    load_resume,
    save_json,
    save_markdown,
    setup_logging,
)

app = Flask(__name__)
logger = logging.getLogger("cvtailro.app")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "cvtailro-dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
app.config["PREFERRED_URL_SCHEME"] = "https"

# Tell Flask it's behind a reverse proxy (Railway/Cloudflare) so url_for
# generates https:// URLs for OAuth callbacks
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# ── CSRF / Session Cookie Hardening ─────────────────────────────────────────
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") != "development"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours


# ── Security Headers ────────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Database
db_url = os.environ.get("DATABASE_URL", "sqlite:///cvtailro_dev.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
if db_url.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {}
else:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 10,
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }

db.init_app(app)
migrate = Migrate(app, db)
login_manager.init_app(app)
init_oauth(app)
app.register_blueprint(auth_bp)
r2_storage.init_app(app)

# Thread-safe job storage
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Clean up jobs older than 15 minutes
JOB_TTL = 900

# Pipeline error log (in-memory, last 50 errors)
pipeline_errors: list[dict] = []
pipeline_errors_lock = threading.Lock()

# Pipeline concurrency control
MAX_CONCURRENT_PIPELINES = 5
pipeline_semaphore = threading.Semaphore(MAX_CONCURRENT_PIPELINES)
pipeline_queue_depth = 0
pipeline_queue_lock = threading.Lock()
MAX_QUEUE_DEPTH = 50


class UsageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = {}
        self._total = 0

    def check_and_record(self, key: str, limit: int) -> bool:
        if limit <= 0:
            with self._lock:
                self._total += 1
            return True
        with self._lock:
            now = time.time()
            times = [t for t in self._requests.get(key, []) if t > now - 3600]
            if len(times) >= limit:
                return False
            times.append(now)
            self._requests[key] = times
            self._total += 1
            return True

    def get_stats(self) -> dict:
        with self._lock:
            now = time.time()
            hour_ago = now - 3600
            active = sum(1 for ts in self._requests.values() if any(t > hour_ago for t in ts))
            recent = sum(len([t for t in ts if t > hour_ago]) for ts in self._requests.values())
            return {"total_requests": self._total, "requests_last_hour": recent, "active_sessions": active}


usage_tracker = UsageTracker()


class LoginRateLimiter:
    """Track failed login attempts by IP to prevent brute-force attacks.

    After MAX_ATTEMPTS failed attempts within WINDOW seconds, the IP is
    blocked for WINDOW seconds.  Successful logins reset the counter.
    """

    MAX_ATTEMPTS = 5
    WINDOW = 900  # 15 minutes

    def __init__(self):
        self._lock = threading.Lock()
        # {ip: [timestamp, ...]}
        self._failures: dict[str, list[float]] = {}

    def is_blocked(self, ip: str) -> bool:
        """Return True if *ip* has exceeded the failure threshold."""
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            self._failures[ip] = attempts
            return len(attempts) >= self.MAX_ATTEMPTS

    def record_failure(self, ip: str) -> None:
        """Record a failed login attempt from *ip*."""
        with self._lock:
            now = time.time()
            attempts = [t for t in self._failures.get(ip, []) if t > now - self.WINDOW]
            attempts.append(now)
            self._failures[ip] = attempts

    def reset(self, ip: str) -> None:
        """Clear failure history for *ip* after a successful login."""
        with self._lock:
            self._failures.pop(ip, None)


login_rate_limiter = LoginRateLimiter()


def _cleanup_old_jobs() -> None:
    """Remove completed jobs older than JOB_TTL seconds."""
    cutoff = time.time() - JOB_TTL
    with jobs_lock:
        expired = [
            k for k, v in jobs.items()
            if v.get("created_at", 0) < cutoff and v.get("status") != "running"
        ]
        for k in expired:
            del jobs[k]


def format_talking_points(talking_points: list) -> str:
    """Format talking points into readable markdown."""
    lines = ["# Interview Talking Points\n"]
    for i, tp in enumerate(talking_points, 1):
        lines.append(f"## {i}. {tp.topic}\n")
        if tp.source_experience:
            lines.append(f"*Based on: {tp.source_experience}*\n")
        for point in tp.bullet_points:
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines)


def run_pipeline_job(
    job_id: str,
    resume_path: str,
    job_text: str,
    mode: str,
    template: str,
    output_dir: Path,
    api_key: str,
    model: str,
    user_id: str | None = None,
) -> None:
    """Run the full pipeline in a background thread, pushing progress events."""
    global pipeline_queue_depth
    progress_queue = jobs[job_id]["queue"]

    def emit(stage: int, total: int, name: str, status: str, detail: str = "") -> None:
        progress_queue.put(
            {
                "stage": stage,
                "total": total,
                "name": name,
                "status": status,
                "detail": detail,
            }
        )

    # Queue management: increment depth, emit queued status, acquire semaphore
    with pipeline_queue_lock:
        pipeline_queue_depth += 1
    emit(0, 7, "Pipeline", "queued", "Waiting for available slot...")
    pipeline_semaphore.acquire()
    with pipeline_queue_lock:
        pipeline_queue_depth -= 1

    try:
        config = AppConfig(
            rewrite_mode=RewriteMode(mode),
            output_dir=str(output_dir),
            api_key=api_key,
            model=model,
            job_id=job_id,
        )

        # Persist TailoringJob to database at the start
        with app.app_context():
            db_job = TailoringJob(
                id=job_id,
                user_id=user_id,
                status="running",
                rewrite_mode=mode,
                template=template,
                model_used=model,
            )
            db.session.add(db_job)
            db.session.commit()

        # Start analytics tracking for this pipeline run
        pipeline_analytics.start_job(job_id, model)

        setup_logging(log_file=output_dir / "pipeline.log")

        pipeline_start = time.time()

        resume_text = load_resume(resume_path)

        # ── Stages 1 + 2 in PARALLEL (independent) ──
        emit(1, 7, "Job Intelligence", "running", "Analysing job description...")
        emit(2, 7, "Resume Parser", "running", "Parsing your resume...")

        job_analysis = None
        resume_data = None
        stage_errors: list[Exception] = []

        def run_stage1():
            nonlocal job_analysis
            try:
                agent1 = JobIntelligenceAgent(config)
                job_analysis = agent1.run(job_text)
                save_json(job_analysis.model_dump(), output_dir / "01_job_analysis.json")
                emit(1, 7, "Job Intelligence", "done",
                     f"{len(job_analysis.required_skills)} required skills, "
                     f"{len(job_analysis.tools)} tools detected")
            except Exception as e:
                stage_errors.append(e)
                emit(1, 7, "Job Intelligence", "error", str(e))

        def run_stage2():
            nonlocal resume_data
            try:
                agent2 = ResumeParserAgent(config)
                resume_data = agent2.run(resume_text)
                save_json(resume_data.model_dump(), output_dir / "02_resume_data.json")
                total_bullets = sum(len(r.bullets) for r in resume_data.roles)
                emit(2, 7, "Resume Parser", "done",
                     f"{len(resume_data.roles)} roles, {total_bullets} bullets, "
                     f"{resume_data.total_years_estimate:.0f} years experience")
            except Exception as e:
                stage_errors.append(e)
                emit(2, 7, "Resume Parser", "error", str(e))

        t1 = threading.Thread(target=run_stage1)
        t2 = threading.Thread(target=run_stage2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if stage_errors:
            raise stage_errors[0]

        logger.info(f"[Pipeline] Stages 1+2 completed in {time.time() - pipeline_start:.1f}s")

        # ── Stage 3 (needs both 1 + 2) ──
        emit(3, 7, "Gap Analysis", "running", "Comparing resume against job...")
        agent3 = GapAnalysisAgent(config)
        gap_report = agent3.run({"job_analysis": job_analysis, "resume_data": resume_data})
        save_json(gap_report.model_dump(), output_dir / "03_gap_report.json")
        emit(3, 7, "Gap Analysis", "done",
             f"Match: {gap_report.match_score:.0f}% | "
             f"Cosine: {gap_report.cosine_similarity:.2f} | "
             f"{len(gap_report.missing_keywords)} missing keywords")

        logger.info(f"[Pipeline] Stage 3 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Stage 4 (needs 2 + 3) ──
        emit(4, 7, "Bullet Optimiser", "running", f"Rewriting bullets ({mode} mode)...")
        agent4 = BulletOptimiserAgent(config)
        optimised_bullets = agent4.run(
            {"resume_data": resume_data, "gap_report": gap_report},
            rewrite_mode=config.rewrite_mode.value,
            job_title=job_analysis.job_title,
            company=job_analysis.company or "the target company",
            job_responsibilities="; ".join(job_analysis.responsibilities[:5]),
            required_skills=", ".join(job_analysis.required_skills[:10]),
        )
        save_json(optimised_bullets.model_dump(), output_dir / "04_optimised_bullets.json")
        fab_count = sum(1 for b in optimised_bullets.bullets if b.fabrication_flag)
        detail = f"{len(optimised_bullets.bullets)} bullets optimised"
        if fab_count:
            detail += f" ({fab_count} flagged)"
        emit(4, 7, "Bullet Optimiser", "done", detail)

        logger.info(f"[Pipeline] Stage 4 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Stages 5 + 6 + 7 in PARALLEL ──
        # ATS, Recruiter, and Talking Points all run concurrently.
        # Talking points use optimised_bullets (available now) instead of
        # waiting for the ATS resume, saving ~60s of wall-clock time.
        emit(5, 7, "ATS Optimiser", "running", "Building ATS-friendly resume...")
        emit(6, 7, "Recruiter Optimiser", "running", "Enhancing for recruiter appeal...")
        emit(7, 7, "Final Assembly", "running", "Generating talking points...")

        ats_resume = None
        recruiter_resume = None
        talking_points = []
        stage_errors.clear()

        def run_stage5():
            nonlocal ats_resume
            try:
                agent5 = ATSOptimiserAgent(config)
                ats_resume = agent5.run({
                    "optimised_bullets": optimised_bullets,
                    "job_analysis": job_analysis,
                    "resume_data": resume_data,
                })
                save_json(ats_resume.model_dump(), output_dir / "05_ats_resume.json")
                checks_passed = sum(1 for c in ats_resume.ats_checks if c.passed)
                emit(5, 7, "ATS Optimiser", "done",
                     f"ATS checks: {checks_passed}/{len(ats_resume.ats_checks)} passed")
            except Exception as e:
                stage_errors.append(e)
                emit(5, 7, "ATS Optimiser", "error", str(e))

        def run_stage6():
            nonlocal recruiter_resume
            try:
                agent6 = RecruiterOptimiserAgent(config)
                recruiter_resume = agent6.run({
                    "optimised_bullets": optimised_bullets,
                    "job_analysis": job_analysis,
                    "gap_report": gap_report,
                    "resume_data": resume_data,
                })
                save_json(recruiter_resume.model_dump(), output_dir / "06_recruiter_resume.json")
                emit(6, 7, "Recruiter Optimiser", "done",
                     f"{len(recruiter_resume.narrative_improvements)} improvements applied")
            except Exception as e:
                stage_errors.append(e)
                emit(6, 7, "Recruiter Optimiser", "error", str(e))

        def run_stage7_talking_points():
            nonlocal talking_points
            try:
                agent7 = FinalAssemblyAgent(config)
                talking_points = agent7._generate_talking_points({
                    "gap_report": gap_report,
                    "job_analysis": job_analysis,
                    "optimised_bullets": optimised_bullets,
                    "resume_data": resume_data,
                })
            except Exception as e:
                logger.warning(f"Talking points generation failed: {e}")
                talking_points = []

        t5 = threading.Thread(target=run_stage5)
        t6 = threading.Thread(target=run_stage6)
        t7 = threading.Thread(target=run_stage7_talking_points)
        t5.start()
        t6.start()
        t7.start()
        t5.join()
        t6.join()
        t7.join()

        if stage_errors:
            raise stage_errors[0]

        logger.info(f"[Pipeline] Stages 5+6+7 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Re-compute match score on the tailored ATS resume ──
        tailored_cos_sim = resume_job_similarity(
            ats_resume.markdown_content,
            job_analysis.raw_text_for_similarity
        )
        # Count how many job keywords appear in the tailored resume
        tailored_text_lower = ats_resume.markdown_content.lower()
        all_job_skills = list(set(
            job_analysis.required_skills + job_analysis.preferred_skills + job_analysis.tools
        ))
        tailored_missing = [s for s in all_job_skills if s.lower() not in tailored_text_lower]
        total_skills = len(all_job_skills)
        tailored_coverage = max(0.0, 1.0 - (len(tailored_missing) / max(total_skills, 1)))
        tailored_match_score = round((tailored_cos_sim * 40) + (tailored_coverage * 40) + 20, 1)
        tailored_match_score = max(0.0, min(100.0, tailored_match_score))

        logger.info(f"[Pipeline] Original score: {gap_report.match_score:.0f}% -> Tailored score: {tailored_match_score:.0f}%")

        # ── Assemble final output (deterministic, instant) ──
        from models import MatchReport
        match_report = MatchReport(
            job_title=job_analysis.job_title,
            company=job_analysis.company,
            overall_match_score=gap_report.match_score,
            cosine_similarity=gap_report.cosine_similarity,
            missing_keywords=gap_report.missing_keywords,
            keyword_frequency=gap_report.keyword_frequency,
            seniority_calibration=gap_report.seniority_calibration,
            ats_checks=ats_resume.ats_checks,
            rewrite_mode=config.rewrite_mode.value,
            optimisation_summary=(
                f"Match score: {gap_report.match_score:.1f}% | "
                f"Cosine similarity: {gap_report.cosine_similarity:.4f} | "
                f"Missing keywords: {len(gap_report.missing_keywords)} | "
                f"Weak alignments: {len(gap_report.weak_alignment)}"
            ),
        )

        # Write artifacts
        save_markdown(ats_resume.markdown_content, output_dir / "tailored_resume_ats.md")
        save_markdown(recruiter_resume.markdown_content, output_dir / "tailored_resume_recruiter.md")
        generate_resume_pdf(ats_resume.markdown_content, output_dir / "tailored_resume_ats.pdf", template=template)
        generate_resume_pdf(recruiter_resume.markdown_content, output_dir / "tailored_resume_recruiter.pdf", template=template)
        generate_resume_docx(ats_resume.markdown_content, output_dir / "tailored_resume_ats.docx")
        generate_resume_docx(recruiter_resume.markdown_content, output_dir / "tailored_resume_recruiter.docx")
        save_json(match_report.model_dump(), output_dir / "match_report.json")
        save_markdown(format_talking_points(talking_points), output_dir / "interview_talking_points.md")

        emit(7, 7, "Final Assembly", "done",
             f"{len(talking_points)} talking points generated")

        total_elapsed = time.time() - pipeline_start
        logger.info(f"[Pipeline] Total pipeline completed in {total_elapsed:.1f}s")

        # Finalise analytics for this pipeline run
        job_stats = pipeline_analytics.complete_job(job_id)
        if job_stats:
            logger.info(
                f"[Pipeline] Analytics: {job_stats.get('total_tokens', 0)} tokens, "
                f"{job_stats.get('api_calls', 0)} API calls, "
                f"${job_stats.get('estimated_cost_usd', 0):.4f} est. cost"
            )

        # Persist results to database and upload files to R2
        files_list = [
            "tailored_resume_ats.pdf",
            "tailored_resume_recruiter.pdf",
            "tailored_resume_ats.docx",
            "tailored_resume_recruiter.docx",
            "tailored_resume_ats.md",
            "tailored_resume_recruiter.md",
            "match_report.json",
            "interview_talking_points.md",
        ]
        with app.app_context():
            db_job = db.session.get(TailoringJob, job_id)
            if db_job:
                db_job.status = "complete"
                db_job.match_score = tailored_match_score
                db_job.original_match_score = gap_report.match_score
                db_job.cosine_similarity = match_report.cosine_similarity
                db_job.missing_keywords = match_report.missing_keywords
                db_job.job_title = job_analysis.job_title
                db_job.company = job_analysis.company
                db_job.ats_resume_md = ats_resume.markdown_content
                db_job.recruiter_resume_md = recruiter_resume.markdown_content
                db_job.talking_points_md = format_talking_points(talking_points)
                db_job.completed_at = datetime.now(timezone.utc)
                db_job.duration_seconds = total_elapsed

                # Upload output files to R2 if configured
                if r2_storage.is_configured:
                    for filename in files_list:
                        file_path = output_dir / filename
                        if file_path.exists():
                            try:
                                r2_key = r2_storage.upload_file(
                                    job_id, filename, file_path=file_path
                                )
                                job_file = JobFile(
                                    job_id=job_id,
                                    filename=filename,
                                    r2_key=r2_key,
                                    content_type=mimetypes.guess_type(filename)[0]
                                    or "application/octet-stream",
                                    size_bytes=file_path.stat().st_size,
                                )
                                db.session.add(job_file)
                            except Exception as upload_err:
                                logger.error(
                                    f"R2 upload failed for {filename}: {upload_err}"
                                )

                db.session.commit()
                logger.info(f"[Pipeline] Job {job_id} persisted to database")

        # Clean up local output files after a delay (give user time to download)
        # Only clean up if R2 upload succeeded
        if r2_storage.is_configured:
            def _cleanup_output(path, delay=300):
                time.sleep(delay)  # Wait 5 minutes before deleting
                shutil.rmtree(str(path), ignore_errors=True)
                logger.info(f"[Pipeline] Cleaned up local output dir (delayed)")
            threading.Thread(target=_cleanup_output, args=(output_dir,), daemon=True).start()

        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = {
                "match_score": match_report.overall_match_score,
                "original_match_score": gap_report.match_score,
                "tailored_match_score": tailored_match_score,
                "cosine_similarity": match_report.cosine_similarity,
                "missing_keywords": match_report.missing_keywords,
                "rewrite_mode": mode,
                "template": template,
                "job_title": job_analysis.job_title,
                "company": job_analysis.company,
                "bullets_rewritten": len(optimised_bullets.bullets),
                "ats_resume_md": ats_resume.markdown_content,
                "recruiter_resume_md": recruiter_resume.markdown_content,
                "talking_points_md": format_talking_points(talking_points),
                "files": [
                    "tailored_resume_ats.pdf",
                    "tailored_resume_recruiter.pdf",
                    "tailored_resume_ats.docx",
                    "tailored_resume_recruiter.docx",
                    "tailored_resume_ats.md",
                    "tailored_resume_recruiter.md",
                    "match_report.json",
                    "interview_talking_points.md",
                ],
            }
        progress_queue.put({"status": "complete"})

        # Reclaim memory from large agent objects
        gc.collect()

        # Schedule delayed cleanup of in-memory result after 2 minutes
        def _delayed_result_cleanup(jid: str, delay: int = 120) -> None:
            time.sleep(delay)
            with jobs_lock:
                if jid in jobs and jobs[jid].get("status") == "complete":
                    jobs[jid].pop("result", None)
                    logger.info(f"[Pipeline] Purged in-memory result for job {jid}")

        threading.Thread(
            target=_delayed_result_cleanup, args=(job_id,), daemon=True
        ).start()

    except Exception as e:
        logging.getLogger("pipeline").exception("Pipeline failed")
        error_msg = str(e)
        # Finalise analytics even on failure (captures partial usage)
        pipeline_analytics.complete_job(job_id)
        # Log to admin-visible error list
        import traceback
        with pipeline_errors_lock:
            pipeline_errors.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "model": model,
                "error": error_msg,
                "traceback": traceback.format_exc()[-500:],
            })
            if len(pipeline_errors) > 50:
                pipeline_errors.pop(0)

        # Mark job as failed in database
        try:
            with app.app_context():
                db_job = db.session.get(TailoringJob, job_id)
                if db_job:
                    db_job.status = "error"
                    db_job.error_message = str(e)[:2000]
                    db_job.completed_at = datetime.now(timezone.utc)
                    db.session.commit()
        except Exception as db_err:
            logger.error(f"Failed to update job status in database: {db_err}")

        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = error_msg
        progress_queue.put({"status": "error", "detail": f"[{model}] {error_msg}"})

    finally:
        pipeline_semaphore.release()


# ─── Admin Routes ────────────────────────────────────────────────────────────────


@app.route("/admin")
def admin_page():
    """Serve the admin panel UI."""
    return render_template("admin.html")


@app.route("/admin/api/login", methods=["POST"])
def admin_login():
    """Authenticate admin or set initial password."""
    client_ip = request.remote_addr or "unknown"

    # ── Brute-force protection ──
    if login_rate_limiter.is_blocked(client_ip):
        logger.warning(f"[Admin] Blocked login attempt from rate-limited IP: {client_ip}")
        return jsonify({"error": "Too many failed attempts. Try again in 15 minutes."}), 429

    data = request.get_json(force=True)
    password = data.get("password", "")
    if not password:
        return jsonify({"error": "Password is required"}), 400

    if not AdminConfigManager.has_password():
        # First time: set the password
        config = AdminConfigManager.load()
        config.admin_password_hash = AdminConfigManager._hash_password(password)
        AdminConfigManager.save(config)
        session["admin_authenticated"] = True
        login_rate_limiter.reset(client_ip)
        logger.info(f"[Admin] Initial password set by IP: {client_ip}")
        return jsonify({"ok": True, "message": "Password set successfully"})
    else:
        # Subsequent: verify
        if AdminConfigManager.verify_password(password):
            session["admin_authenticated"] = True
            login_rate_limiter.reset(client_ip)
            logger.info(f"[Admin] Successful login from IP: {client_ip}")
            return jsonify({"ok": True})
        else:
            login_rate_limiter.record_failure(client_ip)
            logger.warning(f"[Admin] Failed login attempt from IP: {client_ip}")
            return jsonify({"error": "Invalid password"}), 401


@app.route("/admin/api/logout", methods=["POST"])
def admin_logout():
    """Clear admin session."""
    session.pop("admin_authenticated", None)
    return jsonify({"ok": True})


@app.route("/admin/api/config", methods=["GET"])
def admin_get_config():
    """Return current config (API key masked)."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
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


@app.route("/admin/api/config", methods=["POST"])
def admin_save_config():
    """Save admin config."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(force=True)
    config = AdminConfigManager.load()

    client_ip = request.remote_addr or "unknown"
    changes = []
    if "api_key" in data:
        old_has_key = bool(config.api_key)
        config.api_key = data["api_key"]
        new_has_key = bool(config.api_key)
        if old_has_key != new_has_key:
            changes.append(f"api_key={'set' if new_has_key else 'cleared'}")
        elif old_has_key and new_has_key:
            changes.append("api_key=updated")
    if "default_model" in data:
        old_model = config.default_model
        config.default_model = data["default_model"]
        if old_model != config.default_model:
            changes.append(f"default_model={config.default_model}")
    if "allow_user_model_selection" in data:
        config.allow_user_model_selection = bool(data["allow_user_model_selection"])
        changes.append(f"allow_user_model_selection={config.allow_user_model_selection}")
    if "rate_limit_per_hour" in data:
        config.rate_limit_per_hour = int(data["rate_limit_per_hour"])
        changes.append(f"rate_limit_per_hour={config.rate_limit_per_hour}")

    AdminConfigManager.save(config)
    logger.info(
        f"[Admin] Config saved by IP {client_ip}: "
        f"{', '.join(changes) if changes else 'no changes'}"
    )
    return jsonify({"ok": True, "updated_at": config.updated_at})


@app.route("/admin/api/test-key", methods=["POST"])
def admin_test_key():
    """Test an API key against OpenRouter."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
    client_ip = request.remote_addr or "unknown"
    data = request.get_json(force=True)
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"valid": False, "error": "No key provided"})
    # Mask the key for logging (never log the full key)
    masked = f"{api_key[:6]}***" if len(api_key) > 6 else "***"
    try:
        r = http_requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            logger.info(f"[Admin] API key test PASSED by IP {client_ip} (key={masked})")
            return jsonify({"valid": True})
        else:
            logger.info(f"[Admin] API key test FAILED by IP {client_ip} (key={masked}): HTTP {r.status_code}")
            return jsonify({"valid": False, "error": f"HTTP {r.status_code}"})
    except Exception as e:
        logger.warning(f"[Admin] API key test ERROR by IP {client_ip} (key={masked}): {e}")
        return jsonify({"valid": False, "error": str(e)})


@app.route("/admin/api/errors")
def admin_errors():
    """Return recent pipeline errors."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
    with pipeline_errors_lock:
        return jsonify({"errors": list(reversed(pipeline_errors))})


@app.route("/admin/api/usage")
def admin_usage():
    """Return usage statistics."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify(usage_tracker.get_stats())


@app.route("/admin/api/analytics")
def admin_analytics():
    """Return token usage and cost analytics across all pipeline runs."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify(pipeline_analytics.get_global_stats())


@app.route("/admin/api/users")
def admin_users():
    """Return all registered users."""
    if not session.get("admin_authenticated") and not (current_user.is_authenticated and current_user.is_admin):
        return jsonify({"error": "Not authenticated"}), 401
    from database import User
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({
        "total": len(users),
        "users": [{
            "id": u.id, "email": u.email, "name": u.name,
            "picture": u.picture_url, "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login": u.last_login_at.isoformat() if u.last_login_at else None,
            "jobs_count": u.jobs.count(),
        } for u in users],
    })


@app.route("/admin/api/live-stats")
def admin_live_stats():
    """Return live operational stats for monitoring under load."""
    if not session.get("admin_authenticated"):
        return jsonify({"error": "Not authenticated"}), 401

    # Active pipelines = MAX_CONCURRENT - semaphore._value (available slots)
    # threading.Semaphore exposes _value internally
    try:
        available_slots = pipeline_semaphore._value
        active_pipelines = MAX_CONCURRENT_PIPELINES - available_slots
    except AttributeError:
        active_pipelines = -1

    with pipeline_queue_lock:
        current_queue_depth = pipeline_queue_depth

    # Memory usage via resource module (no psutil dependency)
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    import platform
    if platform.system() == "Darwin":
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


@app.route("/api/status")
def api_status():
    """Return whether the service is configured."""
    return jsonify({
        "configured": AdminConfigManager.is_configured(),
        "has_admin_password": AdminConfigManager.has_password(),
    })


# ─── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "backend": "openrouter"})


@app.route("/api/models")
def list_models():
    """Return the curated list of recommended models."""
    config = AdminConfigManager.load()
    default = config.default_model if config.default_model else DEFAULT_MODEL
    return jsonify({
        "models": [
            {"id": model_id, "name": display_name}
            for display_name, model_id in RECOMMENDED_MODELS.items()
        ],
        "default": default,
        "user_selectable": config.allow_user_model_selection,
    })


@app.route("/api/tailor", methods=["POST"])
def start_tailoring():
    """Start a tailoring job. Accepts resume file + job description text."""
    _cleanup_old_jobs()

    # Load API key from admin config
    admin_config = AdminConfigManager.load()
    api_key = admin_config.api_key.strip()

    if not api_key:
        return jsonify({"error": "Service not configured. An admin must set the API key at /admin."}), 400

    # Determine model
    if admin_config.allow_user_model_selection:
        model = request.form.get("model", admin_config.default_model or DEFAULT_MODEL).strip()
    else:
        model = admin_config.default_model or DEFAULT_MODEL

    # Check queue depth before accepting new jobs
    with pipeline_queue_lock:
        if pipeline_queue_depth >= MAX_QUEUE_DEPTH:
            return jsonify({"error": "Server is at capacity. Please try again in a few minutes."}), 503

    # Rate limiting — use user ID if authenticated, IP otherwise
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

    # ── Input sanitization: job description ──
    # Strip HTML tags to prevent stored XSS
    job_text = re.sub(r"<[^>]+>", "", job_text)
    # Enforce max length
    if len(job_text) > 50000:
        return jsonify({"error": "Job description is too long (maximum 50,000 characters)"}), 400

    if not job_text or len(job_text) < 50:
        return jsonify({"error": "Job description is too short (minimum 50 characters)"}), 400
    if mode not in ("conservative", "aggressive"):
        return jsonify({"error": "Invalid mode"}), 400
    if template not in ("executive", "modern", "minimal"):
        return jsonify({"error": "Invalid template"}), 400

    # Validate resume file
    resume_ext = Path(resume_file.filename).suffix.lower()
    if resume_ext == ".pdf":
        # ── Validate PDF magic bytes (%PDF-) ──
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
                    return jsonify({"error": "PDF appears to be image-based or empty. Please use a text-based PDF."}), 400
            resume_file.stream.seek(0)
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {e}"}), 400
    elif resume_ext not in (".md", ".txt"):
        return jsonify({"error": "Unsupported file type. Use PDF, MD, or TXT."}), 400

    # Log incoming request metadata
    logger.info(
        f"[/api/tailor] New request: client_ip={client_ip}, model={model}, "
        f"mode={mode}, template={template}, jd_length={len(job_text)}, "
        f"resume_file={resume_file.filename}"
    )

    # Create job
    job_id = uuid.uuid4().hex[:16]
    output_dir = create_output_dir()

    # Save uploaded resume
    resume_path = output_dir / f"input_resume{resume_ext}"
    resume_file.save(str(resume_path))

    # Save job description
    (output_dir / "input_job_description.txt").write_text(job_text, encoding="utf-8")

    # Get user_id if authenticated (not available in background thread)
    user_id = current_user.id if current_user.is_authenticated else None

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "queue": queue.Queue(),
            "output_dir": str(output_dir),
            "created_at": time.time(),
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, str(resume_path), job_text, mode, template, output_dir,
              api_key, model, user_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    """SSE endpoint for real-time pipeline progress."""
    with jobs_lock:
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
                # Check if the job still exists (may have been cleaned up)
                with jobs_lock:
                    if job_id not in jobs:
                        yield f"data: {json.dumps({'status': 'error', 'detail': 'Job expired'})}\n\n"
                        return
                # Send keepalive every 10s to prevent Railway/proxy timeout
                yield f"data: {json.dumps({'status': 'keepalive'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/result/<job_id>")
def get_result(job_id: str):
    """Get the final result of a completed job.

    Checks in-memory jobs first, then falls back to the database for
    jobs that have been purged from memory but persist in the DB.
    """
    with jobs_lock:
        job = jobs.get(job_id)

    if job is not None:
        if job["status"] == "running":
            return jsonify({"status": "running"}), 202
        if job["status"] == "error":
            return jsonify({"status": "error", "error": job["error"]}), 500
        # Only return in-memory result if it hasn't been purged
        if job.get("result") is not None:
            return jsonify({"status": "complete", "result": job["result"]})
        # Result was purged from memory — fall through to DB

    # Fallback: query the database
    db_job = db.session.get(TailoringJob, job_id)
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
        "files": [f.filename for f in db_job.files],
    }
    if db_job.original_match_score is not None:
        result["original_match_score"] = db_job.original_match_score
    if db_job.match_score is not None:
        result["tailored_match_score"] = db_job.match_score
    return jsonify({"status": "complete", "result": result})


@app.route("/api/download/<job_id>/<filename>")
def download_file(job_id: str, filename: str):
    """Download an output file from a completed job.

    Three fallback paths:
      1. In-memory jobs (current session, files on disk)
      2. R2 cloud storage (if configured, via presigned URL)
      3. Database (markdown served directly, PDFs regenerated on-the-fly)
    """
    safe_name = Path(filename).name
    is_authed = current_user.is_authenticated
    user_id = current_user.id if is_authed else None
    logger.info(
        f"[Download] job_id={job_id} file={safe_name} "
        f"authed={is_authed} user_id={user_id}"
    )

    # 1. Check in-memory jobs (current session)
    output_dir = None
    with jobs_lock:
        if job_id in jobs:
            output_dir = jobs[job_id]["output_dir"]

    if output_dir:
        file_path = Path(output_dir) / safe_name
        if file_path.exists():
            logger.info(f"[Download] Serving from local disk: {file_path}")
            return send_from_directory(output_dir, safe_name, as_attachment=True)
        else:
            logger.info(f"[Download] In-memory job found but file missing on disk: {file_path}")

    # 2. Fall back to R2 for historical files
    if r2_storage.is_configured:
        job_file = JobFile.query.filter_by(
            job_id=job_id, filename=safe_name
        ).first()
        if job_file:
            # Verify the job belongs to the current user (if authenticated)
            if is_authed:
                job_record = TailoringJob.query.filter_by(
                    id=job_id, user_id=user_id
                ).first()
                if not job_record:
                    logger.warning(f"[Download] R2 access denied: job {job_id} user {user_id}")
                    return jsonify({"error": "Access denied"}), 403
            try:
                url = r2_storage.generate_presigned_url(job_file.r2_key)
                logger.info(f"[Download] Redirecting to R2 presigned URL for {safe_name}")
                return redirect(url)
            except Exception as e:
                logger.error(f"[Download] R2 presigned URL failed: {e}")
                # Fall through to DB path instead of returning error
        else:
            logger.info(f"[Download] R2 configured but no JobFile record for {safe_name}")

    # 3. Serve from DB (markdown directly, PDFs regenerated, JSON reconstructed)
    # Try with user_id filter first, then without (handles jobs created before auth)
    db_job = None
    if is_authed:
        db_job = TailoringJob.query.filter_by(id=job_id, user_id=user_id).first()
    if db_job is None:
        # Fallback: try without user filter (for jobs with null user_id or
        # created by the same user before session rotation)
        db_job = db.session.get(TailoringJob, job_id)
        if db_job and db_job.user_id is not None and db_job.user_id != user_id:
            # Job belongs to a different user — deny access
            logger.warning(
                f"[Download] DB access denied: job {job_id} belongs to "
                f"user {db_job.user_id}, requester is {user_id}"
            )
            db_job = None

    if db_job is None:
        logger.warning(
            f"[Download] Job {job_id} not found in DB "
            f"(user_id={user_id})"
        )
        return jsonify({"error": "File not found. The job may have expired or you may need to sign in again."}), 404

    logger.info(
        f"[Download] Found job in DB: status={db_job.status} "
        f"has_ats_md={db_job.ats_resume_md is not None} "
        f"has_rec_md={db_job.recruiter_resume_md is not None} "
        f"has_tp_md={db_job.talking_points_md is not None}"
    )

    from flask import Response as FlaskResponse

    # Markdown files — serve directly from DB
    md_content = None
    if "ats" in safe_name and safe_name.endswith(".md"):
        md_content = db_job.ats_resume_md
    elif "recruiter" in safe_name and safe_name.endswith(".md"):
        md_content = db_job.recruiter_resume_md
    elif "talking" in safe_name and safe_name.endswith(".md"):
        md_content = db_job.talking_points_md

    if md_content:
        logger.info(f"[Download] Serving markdown from DB: {safe_name} ({len(md_content)} chars)")
        return FlaskResponse(
            md_content, mimetype="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={safe_name}"}
        )
    elif safe_name.endswith(".md"):
        logger.warning(f"[Download] Markdown requested but content is NULL in DB for {safe_name}")
        return jsonify({"error": f"Resume content not found in database. The pipeline may not have saved results for this job."}), 404

    # PDF files — regenerate from stored markdown
    if safe_name.endswith(".pdf"):
        source_md = None
        if "ats" in safe_name:
            source_md = db_job.ats_resume_md
        elif "recruiter" in safe_name:
            source_md = db_job.recruiter_resume_md

        if not source_md:
            logger.warning(f"[Download] PDF requested but source markdown is NULL for {safe_name}")
            return jsonify({"error": "Resume content not saved. Try re-running the pipeline."}), 404

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
            template = db_job.template or "modern"
            logger.info(f"[Download] Regenerating PDF from DB markdown: {safe_name} template={template}")
            generate_resume_pdf(source_md, tmp_path, template=template)
            pdf_data = open(tmp_path, "rb").read()
            os.unlink(tmp_path)
            return FlaskResponse(
                pdf_data, mimetype="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={safe_name}"}
            )
        except Exception as e:
            logger.error(f"[Download] PDF regeneration failed for {safe_name}: {e}", exc_info=True)
            return jsonify({"error": "Could not generate PDF. Try downloading the markdown version instead."}), 500

    # DOCX files — regenerate from stored markdown
    if safe_name.endswith(".docx"):
        source_md = None
        if "ats" in safe_name:
            source_md = db_job.ats_resume_md
        elif "recruiter" in safe_name:
            source_md = db_job.recruiter_resume_md

        if not source_md:
            logger.warning(f"[Download] DOCX requested but source markdown is NULL for {safe_name}")
            return jsonify({"error": "Resume content not saved. Try re-running the pipeline."}), 404

        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp_path = tmp.name
            logger.info(f"[Download] Regenerating DOCX from DB markdown: {safe_name}")
            generate_resume_docx(source_md, tmp_path)
            docx_data = open(tmp_path, "rb").read()
            os.unlink(tmp_path)
            return FlaskResponse(
                docx_data,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={safe_name}"}
            )
        except Exception as e:
            logger.error(f"[Download] DOCX regeneration failed for {safe_name}: {e}", exc_info=True)
            return jsonify({"error": "Could not generate DOCX. Try downloading the markdown version instead."}), 500

    # Match report JSON — reconstruct from stored fields
    if "match_report" in safe_name and safe_name.endswith(".json"):
        import json as json_mod
        report = {
            "job_title": db_job.job_title,
            "company": db_job.company,
            "match_score": db_job.match_score,
            "cosine_similarity": db_job.cosine_similarity,
            "missing_keywords": db_job.missing_keywords,
        }
        logger.info(f"[Download] Serving reconstructed match report JSON")
        return FlaskResponse(
            json_mod.dumps(report, indent=2), mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={safe_name}"}
        )

    logger.warning(f"[Download] Unrecognised filename pattern: {safe_name}")
    return jsonify({"error": "File not found. Try re-downloading from your History."}), 404


@app.route("/api/download-check/<job_id>")
@login_required
def download_check(job_id: str):
    """Diagnostic endpoint: check what download paths are available for a job.

    Useful for debugging download failures from the browser console:
        fetch('/api/download-check/<job_id>').then(r => r.json()).then(console.log)
    """
    result = {
        "job_id": job_id,
        "user_id": current_user.id,
        "in_memory": False,
        "local_files": [],
        "r2_configured": r2_storage.is_configured,
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
                result["local_files"] = [
                    f.name for f in Path(output_dir).iterdir() if f.is_file()
                ]
            except Exception:
                pass

    if r2_storage.is_configured:
        job_files = JobFile.query.filter_by(job_id=job_id).all()
        result["r2_files"] = [jf.filename for jf in job_files]

    db_job = db.session.get(TailoringJob, job_id)
    if db_job:
        result["db_found"] = True
        result["db_status"] = db_job.status
        result["db_has_ats_md"] = db_job.ats_resume_md is not None and len(db_job.ats_resume_md or "") > 0
        result["db_has_rec_md"] = db_job.recruiter_resume_md is not None and len(db_job.recruiter_resume_md or "") > 0
        result["db_has_tp_md"] = db_job.talking_points_md is not None and len(db_job.talking_points_md or "") > 0
        result["db_user_id"] = db_job.user_id
        result["db_user_match"] = db_job.user_id == current_user.id

    return jsonify(result)


# ─── History API ─────────────────────────────────────────────────────────────


@app.route("/api/history")
@login_required
def get_history():
    """Return paginated job history for the current user."""
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 50)
    query = TailoringJob.query.filter_by(user_id=current_user.id).order_by(
        TailoringJob.created_at.desc()
    )
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        "jobs": [{
            "id": j.id,
            "status": j.status,
            "job_title": j.job_title,
            "company": j.company,
            "match_score": j.match_score,
            "rewrite_mode": j.rewrite_mode,
            "model_used": j.model_used,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "files": [
                {"filename": f.filename, "size_bytes": f.size_bytes}
                for f in j.files
            ],
        } for j in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
    })


@app.route("/api/history/<job_id>")
@login_required
def get_history_job(job_id):
    """Return full details of a specific job for the current user."""
    job = TailoringJob.query.filter_by(
        id=job_id, user_id=current_user.id
    ).first_or_404()
    return jsonify({
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
        "model_used": job.model_used,
        "ats_resume_md": job.ats_resume_md,
        "recruiter_resume_md": job.recruiter_resume_md,
        "talking_points_md": job.talking_points_md,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "files": [
            {"filename": f.filename, "size_bytes": f.size_bytes}
            for f in job.files
        ],
    })


if __name__ == "__main__":
    setup_logging(verbose=False)
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  CVtailro Web UI\n  ───────────────\n  Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
