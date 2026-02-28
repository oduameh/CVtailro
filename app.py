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

import json
import logging
import mimetypes
import os
import queue
import re
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
        "pool_size": 5,
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

# Clean up jobs older than 2 hours
JOB_TTL = 7200

# Pipeline error log (in-memory, last 50 errors)
pipeline_errors: list[dict] = []
pipeline_errors_lock = threading.Lock()


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
            "tailored_resume_ats.md",
            "tailored_resume_recruiter.md",
            "match_report.json",
            "interview_talking_points.md",
        ]
        with app.app_context():
            db_job = db.session.get(TailoringJob, job_id)
            if db_job:
                db_job.status = "complete"
                db_job.match_score = match_report.overall_match_score
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

        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = {
                "match_score": match_report.overall_match_score,
                "cosine_similarity": match_report.cosine_similarity,
                "missing_keywords": match_report.missing_keywords,
                "rewrite_mode": mode,
                "template": template,
                "ats_resume_md": ats_resume.markdown_content,
                "recruiter_resume_md": recruiter_resume.markdown_content,
                "talking_points_md": format_talking_points(talking_points),
                "files": [
                    "tailored_resume_ats.pdf",
                    "tailored_resume_recruiter.pdf",
                    "tailored_resume_ats.md",
                    "tailored_resume_recruiter.md",
                    "match_report.json",
                    "interview_talking_points.md",
                ],
            }
        progress_queue.put({"status": "complete"})

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

    # Rate limiting
    client_ip = request.remote_addr or "unknown"
    if not usage_tracker.check_and_record(client_ip, admin_config.rate_limit_per_hour):
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
    """Get the final result of a completed job."""
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        job = jobs[job_id]

    if job["status"] == "running":
        return jsonify({"status": "running"}), 202
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job["error"]}), 500
    return jsonify({"status": "complete", "result": job["result"]})


@app.route("/api/download/<job_id>/<filename>")
def download_file(job_id: str, filename: str):
    """Download an output file from a completed job.

    Checks in-memory jobs first (current session), then falls back to R2
    storage for historical files.
    """
    safe_name = Path(filename).name

    # 1. Check in-memory jobs (current session)
    with jobs_lock:
        if job_id in jobs:
            output_dir = jobs[job_id]["output_dir"]
            file_path = Path(output_dir) / safe_name
            if file_path.exists():
                return send_from_directory(output_dir, safe_name, as_attachment=True)

    # 2. Fall back to R2 for historical files
    if r2_storage.is_configured:
        job_file = JobFile.query.filter_by(
            job_id=job_id, filename=safe_name
        ).first()
        if job_file:
            # Verify the job belongs to the current user (if authenticated)
            if current_user.is_authenticated:
                job_record = TailoringJob.query.filter_by(
                    id=job_id, user_id=current_user.id
                ).first()
                if not job_record:
                    return jsonify({"error": "Access denied"}), 403
            try:
                url = r2_storage.generate_presigned_url(job_file.r2_key)
                return redirect(url)
            except Exception as e:
                logger.error(f"R2 presigned URL failed: {e}")
                return jsonify({"error": "File temporarily unavailable"}), 500

    return jsonify({"error": "Job not found"}), 404


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
