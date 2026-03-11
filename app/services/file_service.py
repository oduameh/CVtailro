"""File download service — serves files from local disk, R2, or DB fallback.

Three-tier download strategy:
  1. In-memory jobs → local disk
  2. R2 cloud storage → presigned URL redirect
  3. Database → regenerate PDF/DOCX from stored markdown
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from flask import Response, jsonify, redirect, send_from_directory
from flask_login import current_user

from app.extensions import db
from app.models import JobFile, TailoringJob
from app.services.pipeline import jobs, jobs_lock
from app.services.telemetry import track
from docx_generator import generate_resume_docx
from pdf_generator import generate_resume_pdf
from storage import r2_storage

logger = logging.getLogger("cvtailro.downloads")


def _content_disposition(filename: str) -> str:
    """RFC 6266-compliant Content-Disposition header value."""
    ascii_name = filename.encode("ascii", "replace").decode()
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def serve_download(job_id: str, filename: str):
    """Serve a file for the given job, trying local → R2 → DB in order."""
    safe_name = Path(filename).name

    ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".json"}
    if Path(safe_name).suffix.lower() not in ALLOWED_SUFFIXES:
        return jsonify({"error": "File type not allowed"}), 403

    BLOCKED_PREFIXES = ("input_", "pipeline")
    if safe_name.lower().startswith(BLOCKED_PREFIXES):
        return jsonify({"error": "File not available for download"}), 403

    is_authed = current_user.is_authenticated
    user_id = current_user.id if is_authed else None

    # 1. In-memory jobs (current session, files on disk)
    output_dir = None
    job_user_id = None
    with jobs_lock:
        if job_id in jobs:
            output_dir = jobs[job_id]["output_dir"]
            job_user_id = jobs[job_id].get("user_id")

    if output_dir:
        if not _can_access_job(job_user_id, user_id, is_authed):
            track("download.failed", category="download", job_id=job_id, metadata={"reason": "access_denied", "source": "local"})
            return jsonify({"error": "Access denied"}), 403
        file_path = Path(output_dir) / safe_name
        if file_path.exists():
            track("download.served", category="download", user_id=user_id, job_id=job_id, metadata={"source": "local", "ext": Path(safe_name).suffix})
            return send_from_directory(output_dir, safe_name, as_attachment=True)

    # 2. R2 cloud storage
    if r2_storage.is_configured:
        job_file = JobFile.query.filter_by(job_id=job_id, filename=safe_name).first()
        if job_file:
            job_record = TailoringJob.query.filter_by(id=job_id).first()
            if not job_record or not _can_access_job(job_record.user_id, user_id, is_authed):
                return jsonify({"error": "Access denied"}), 403
            try:
                url = r2_storage.generate_presigned_url(job_file.r2_key)
                track("download.served", category="download", user_id=user_id, job_id=job_id, metadata={"source": "r2", "ext": Path(safe_name).suffix})
                return redirect(url)
            except Exception as e:
                logger.error(f"R2 presigned URL failed: {e}")

    # 3. Database fallback
    db_job = _resolve_job_ownership(job_id, user_id, is_authed)
    if db_job is None:
        track("download.failed", category="download", job_id=job_id, metadata={"reason": "not_found"})
        return (
            jsonify({"error": "File not found. The job may have expired or you may need to sign in again."}),
            404,
        )

    track("download.served", category="download", user_id=user_id, job_id=job_id, metadata={"source": "db_regen", "ext": Path(safe_name).suffix})
    return _serve_from_db(db_job, safe_name)


def _can_access_job(job_user_id: str | None, request_user_id: str | None, is_authed: bool) -> bool:
    """Return True if the current user may access a job (owned by job_user_id)."""
    if job_user_id is None:
        return True  # Anonymous job — anyone can access
    if not is_authed:
        return False  # Authenticated user's job — anonymous cannot access
    return job_user_id == request_user_id


def _resolve_job_ownership(job_id: str, user_id: str | None, is_authed: bool) -> TailoringJob | None:
    """Find a TailoringJob the current user is allowed to access."""
    db_job = None
    if is_authed:
        db_job = TailoringJob.query.filter_by(id=job_id, user_id=user_id).first()
    if db_job is None:
        db_job = db.session.get(TailoringJob, job_id)
        if db_job and db_job.user_id is not None and db_job.user_id != user_id:
            db_job = None
    return db_job


def _is_recruiter_file(name: str) -> bool:
    return "recruiter" in name.lower()


def _pick_source_md(db_job: TailoringJob, filename: str) -> str | None:
    """Return the right markdown source: recruiter (stripped) or full ATS."""
    if _is_recruiter_file(filename) and db_job.recruiter_resume_md:
        return db_job.recruiter_resume_md
    return db_job.ats_resume_md


def _serve_from_db(db_job: TailoringJob, safe_name: str):
    """Serve a file by regenerating it from data stored in the database."""

    # Markdown files
    if safe_name.endswith(".md"):
        md_content = None
        lower_name = safe_name.lower()
        is_resume = (
            "ats" in lower_name
            or "recruiter" in lower_name
            or ("resume" in lower_name and "talking" not in lower_name)
        )
        if is_resume:
            md_content = _pick_source_md(db_job, safe_name)
        elif "talking" in lower_name:
            md_content = db_job.talking_points_md

        if md_content:
            return Response(
                md_content,
                mimetype="text/markdown",
                headers={"Content-Disposition": _content_disposition(safe_name)},
            )
        return jsonify({"error": "Resume content not found in database."}), 404

    # PDF files — regenerate from stored markdown
    if safe_name.endswith(".pdf"):
        if "cover" in safe_name.lower():
            source_md = db_job.cover_letter_md
            if not source_md:
                return jsonify({"error": "No cover letter available for this job."}), 404
            return _regenerate_pdf(source_md, safe_name, "modern")

        lower_safe = safe_name.lower()
        all_templates = (
            "executive",
            "modern",
            "minimal",
            "creative",
            "compact",
            "professional",
            "tech",
            "elegant",
        )
        is_resume_pdf = any(k in lower_safe for k in ("ats", "recruiter", "resume") + all_templates)
        source_md = _pick_source_md(db_job, safe_name)
        if is_resume_pdf and source_md:
            template = db_job.template or "modern"
            for tpl in all_templates:
                if tpl in lower_safe:
                    template = tpl
                    break
            return _regenerate_pdf(source_md, safe_name, template)

        return jsonify({"error": "Resume content not saved. Try re-running the pipeline."}), 404

    # DOCX files
    if safe_name.endswith(".docx"):
        source_md = _pick_source_md(db_job, safe_name)
        if source_md:
            return _regenerate_docx(source_md, safe_name)
        return jsonify({"error": "Resume content not saved. Try re-running the pipeline."}), 404

    # Match report JSON (case-insensitive: Match_Report.json, match_report.json, etc.)
    if "match_report" in safe_name.lower() and safe_name.lower().endswith(".json"):
        report = {
            "job_title": db_job.job_title,
            "company": db_job.company,
            "match_score": db_job.match_score,
            "cosine_similarity": db_job.cosine_similarity,
            "missing_keywords": db_job.missing_keywords,
        }
        return Response(
            json.dumps(report, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": _content_disposition(safe_name)},
        )

    return jsonify({"error": "File not found. Try re-downloading from your History."}), 404


def _regenerate_pdf(source_md: str, filename: str, template: str):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        generate_resume_pdf(source_md, tmp_path, template=template)
        with open(tmp_path, "rb") as f:
            pdf_data = f.read()
        return Response(
            pdf_data,
            mimetype="application/pdf",
            headers={"Content-Disposition": _content_disposition(filename)},
        )
    except Exception as e:
        logger.error(f"PDF regeneration failed for {filename}: {e}", exc_info=True)
        return jsonify({"error": "Could not generate PDF."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _regenerate_docx(source_md: str, filename: str):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp_path = tmp.name
        all_templates = (
            "executive", "modern", "minimal", "creative",
            "compact", "professional", "tech", "elegant",
        )
        template = "modern"
        lower_name = filename.lower()
        for tpl in all_templates:
            if tpl in lower_name:
                template = tpl
                break
        generate_resume_docx(source_md, tmp_path, template=template)
        with open(tmp_path, "rb") as f:
            docx_data = f.read()
        return Response(
            docx_data,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": _content_disposition(filename)},
        )
    except Exception as e:
        logger.error(f"DOCX regeneration failed for {filename}: {e}", exc_info=True)
        return jsonify({"error": "Could not generate DOCX."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
