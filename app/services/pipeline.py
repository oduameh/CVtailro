"""Pipeline orchestration — runs the 6-stage resume tailoring pipeline.

Extracted from the monolithic app.py. Manages concurrency via semaphore,
emits progress events through a queue, and persists results to the DB.
"""

from __future__ import annotations

import gc
import logging
import mimetypes
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask

from agents.bullet_optimiser import BulletOptimiserAgent
from agents.final_assembly import FinalAssemblyAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.job_intelligence import JobIntelligenceAgent
from agents.resume_optimiser import ResumeOptimiserAgent
from agents.resume_parser import ResumeParserAgent
from analytics import pipeline_analytics
from app.extensions import db
from app.models import JobFile, TailoringJob
from app.services.telemetry import track_with_app
from config import AppConfig
from docx_generator import generate_resume_docx
from models import MatchReport, RewriteMode
from pdf_generator import ALL_TEMPLATE_NAMES, generate_resume_pdf
from similarity import resume_job_similarity
from storage import r2_storage
from utils import load_resume, save_json, save_markdown, setup_logging

logger = logging.getLogger("cvtailro.pipeline")

# ── Thread-safe job storage ──────────────────────────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Pipeline error log (in-memory, last 50 errors)
pipeline_errors: list[dict] = []
pipeline_errors_lock = threading.Lock()

# Pipeline concurrency — configurable via app settings
MAX_CONCURRENT_PIPELINES = 5
pipeline_semaphore = threading.Semaphore(MAX_CONCURRENT_PIPELINES)
pipeline_queue_depth = 0
pipeline_queue_lock = threading.Lock()
MAX_QUEUE_DEPTH = 50


def cleanup_old_jobs(ttl: int = 900) -> None:
    """Remove completed jobs older than *ttl* seconds from in-memory storage."""
    cutoff = time.time() - ttl
    with jobs_lock:
        expired = [
            k for k, v in jobs.items() if v.get("created_at", 0) < cutoff and v.get("status") != "running"
        ]
        for k in expired:
            del jobs[k]


def format_talking_points(talking_points: list) -> str:
    lines = ["# Interview Talking Points\n"]
    for i, tp in enumerate(talking_points, 1):
        lines.append(f"## {i}. {tp.topic}\n")
        if tp.source_experience:
            lines.append(f"*Based on: {tp.source_experience}*\n")
        for point in tp.bullet_points:
            lines.append(f"- {point}")
        lines.append("")
    return "\n".join(lines)


def safe_filename(job_title: str | None, company: str | None, suffix: str) -> str:
    title = re.sub(r"[^\w\s-]", "", job_title or "Resume")[:30].strip()
    comp = re.sub(r"[^\w\s-]", "", company or "")[:20].strip()
    title = title.replace(" ", "_")
    comp = comp.replace(" ", "_")
    return f"{title}_{comp}_{suffix}" if comp else f"{title}_{suffix}"


def run_pipeline_job(
    flask_app: Flask,
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

    def emit(
        stage: int,
        total: int,
        name: str,
        status: str,
        detail: str = "",
        position: int | None = None,
    ) -> None:
        payload = {"stage": stage, "total": total, "name": name, "status": status, "detail": detail}
        if position is not None:
            payload["position"] = position
        progress_queue.put(payload)

    def _te(name, **kw):
        track_with_app(flask_app, name, category="tailor", user_id=user_id, job_id=job_id, **kw)

    with pipeline_queue_lock:
        pipeline_queue_depth += 1
        queue_position = pipeline_queue_depth
    queue_enter = time.time()
    emit(0, 6, "Pipeline", "queued", "Waiting for available slot...", position=queue_position)
    pipeline_semaphore.acquire()
    with pipeline_queue_lock:
        pipeline_queue_depth -= 1
    queue_wait_ms = round((time.time() - queue_enter) * 1000)
    _te("tailor.queue.exited", metadata={"wait_ms": queue_wait_ms, "model": model})

    log_cleanup = None
    try:
        pipeline_config = AppConfig(
            rewrite_mode=RewriteMode(mode),
            output_dir=str(output_dir),
            api_key=api_key,
            model=model,
            job_id=job_id,
        )

        with flask_app.app_context():
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

        pipeline_analytics.start_job(job_id, model)
        log_cleanup = setup_logging(log_file=output_dir / "pipeline.log")
        pipeline_start = time.time()

        resume_text = load_resume(resume_path)

        with flask_app.app_context():
            db_job = db.session.get(TailoringJob, job_id)
            if db_job:
                db_job.original_resume_text = resume_text
                db_job.job_description_full = job_text
                db.session.commit()

        # ── Stages 1 + 2 PARALLEL ────────────────────────────────────────────
        emit(1, 6, "Job Intelligence", "running", "Analysing job description...")
        emit(2, 6, "Resume Parser", "running", "Parsing your resume...")

        job_analysis = None
        resume_data = None
        stage_errors: list[Exception] = []

        def run_stage1():
            nonlocal job_analysis
            try:
                agent1 = JobIntelligenceAgent(pipeline_config)
                job_analysis = agent1.run(job_text)
                save_json(job_analysis.model_dump(), output_dir / "01_job_analysis.json")
                emit(
                    1,
                    6,
                    "Job Intelligence",
                    "done",
                    f"{len(job_analysis.required_skills)} required skills, "
                    f"{len(job_analysis.tools)} tools detected",
                )
            except Exception as e:
                stage_errors.append(e)
                emit(1, 6, "Job Intelligence", "error", str(e))

        def run_stage2():
            nonlocal resume_data
            try:
                agent2 = ResumeParserAgent(pipeline_config)
                resume_data = agent2.run(resume_text)
                save_json(resume_data.model_dump(), output_dir / "02_resume_data.json")
                total_bullets = sum(len(r.bullets) for r in resume_data.roles)
                emit(
                    2,
                    6,
                    "Resume Parser",
                    "done",
                    f"{len(resume_data.roles)} roles, {total_bullets} bullets, "
                    f"{resume_data.total_years_estimate:.0f} years experience",
                )
            except Exception as e:
                stage_errors.append(e)
                emit(2, 6, "Resume Parser", "error", str(e))

        t1 = threading.Thread(target=run_stage1)
        t2 = threading.Thread(target=run_stage2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if stage_errors:
            raise stage_errors[0]
        logger.info(f"Stages 1+2 completed in {time.time() - pipeline_start:.1f}s")

        # ── Stage 3 ──────────────────────────────────────────────────────────
        emit(3, 6, "Gap Analysis", "running", "Comparing resume against job...")
        agent3 = GapAnalysisAgent(pipeline_config)
        gap_report = agent3.run({"job_analysis": job_analysis, "resume_data": resume_data})
        save_json(gap_report.model_dump(), output_dir / "03_gap_report.json")
        emit(
            3,
            6,
            "Gap Analysis",
            "done",
            f"Match: {gap_report.match_score:.0f}% | "
            f"Cosine: {gap_report.cosine_similarity:.2f} | "
            f"{len(gap_report.missing_keywords)} missing keywords",
        )
        logger.info(f"Stage 3 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Stage 4 ──────────────────────────────────────────────────────────
        emit(4, 6, "Bullet Optimiser", "running", f"Rewriting bullets ({mode} mode)...")
        agent4 = BulletOptimiserAgent(pipeline_config)
        optimised_bullets = agent4.run(
            {"resume_data": resume_data, "gap_report": gap_report},
            rewrite_mode=pipeline_config.rewrite_mode.value,
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
        emit(4, 6, "Bullet Optimiser", "done", detail)
        logger.info(f"Stage 4 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Stages 5 + 6 PARALLEL ────────────────────────────────────────────
        emit(5, 6, "Resume Optimiser", "running", "Building your optimised resume...")
        emit(6, 6, "Final Assembly", "running", "Generating talking points...")

        ats_resume = None
        talking_points: list = []
        stage_errors.clear()

        def run_stage5():
            nonlocal ats_resume
            try:
                agent5 = ResumeOptimiserAgent(pipeline_config)
                ats_resume = agent5.run(
                    {
                        "optimised_bullets": optimised_bullets,
                        "job_analysis": job_analysis,
                        "gap_report": gap_report,
                        "resume_data": resume_data,
                    }
                )
                save_json(ats_resume.model_dump(), output_dir / "05_ats_resume.json")
                checks_passed = sum(1 for c in ats_resume.ats_checks if c.passed)
                emit(
                    5,
                    6,
                    "Resume Optimiser",
                    "done",
                    f"ATS checks: {checks_passed}/{len(ats_resume.ats_checks)} passed",
                )
            except Exception as e:
                stage_errors.append(e)
                emit(5, 6, "Resume Optimiser", "error", str(e))

        def run_stage6():
            nonlocal talking_points
            try:
                agent6 = FinalAssemblyAgent(pipeline_config)
                talking_points = agent6._generate_talking_points(
                    {
                        "gap_report": gap_report,
                        "job_analysis": job_analysis,
                        "optimised_bullets": optimised_bullets,
                        "resume_data": resume_data,
                    }
                )
            except Exception as e:
                logger.warning(f"Talking points generation failed: {e}")
                talking_points = []

        t5 = threading.Thread(target=run_stage5)
        t6 = threading.Thread(target=run_stage6)
        t5.start()
        t6.start()
        t5.join()
        t6.join()

        if stage_errors:
            raise stage_errors[0]
        logger.info(f"Stages 5+6 completed at {time.time() - pipeline_start:.1f}s elapsed")

        # ── Re-compute match score on tailored resume ────────────────────────
        tailored_cos_sim = resume_job_similarity(
            ats_resume.markdown_content, job_analysis.raw_text_for_similarity
        )
        tailored_text_lower = ats_resume.markdown_content.lower()
        all_job_skills = list(
            set(job_analysis.required_skills + job_analysis.preferred_skills + job_analysis.tools)
        )
        tailored_found = 0
        tailored_missing = []
        for s in all_job_skills:
            s_lower = s.lower().strip()
            if not s_lower:
                continue
            if len(s_lower.split()) > 1:
                if s_lower in tailored_text_lower:
                    tailored_found += 1
                else:
                    tailored_missing.append(s)
            else:
                if re.search(r"\b" + re.escape(s_lower) + r"\b", tailored_text_lower):
                    tailored_found += 1
                else:
                    tailored_missing.append(s)
        total_skills = len(all_job_skills) or 1
        tailored_coverage = tailored_found / total_skills
        scaled_cos = min(tailored_cos_sim * 2.5, 1.0)
        tailored_match_score = round((scaled_cos * 25) + (tailored_coverage * 60) + 15, 1)
        tailored_match_score = max(0.0, min(100.0, tailored_match_score))
        logger.info(
            f"Original score: {gap_report.match_score:.0f}% -> Tailored score: {tailored_match_score:.0f}%"
        )

        # ── Recompute section scores against tailored resume ─────────────────
        tailored_section_scores: dict = {}
        try:
            sections = {"summary": "", "experience": "", "skills": "", "education": ""}
            current_section = ""
            for line in ats_resume.markdown_content.split("\n"):
                stripped = line.strip().lower()
                if stripped.startswith("#"):
                    heading = stripped.lstrip("#").strip().strip("*").strip()
                    if not heading or "|" in line:
                        continue
                    if any(k in heading for k in ("summary", "profile", "objective")):
                        current_section = "summary"
                    elif any(k in heading for k in ("experience", "employment", "work")):
                        current_section = "experience"
                    elif "skill" in heading:
                        current_section = "skills"
                    elif "education" in heading:
                        current_section = "education"
                    else:
                        current_section = ""
                elif current_section:
                    sections[current_section] += " " + line

            for sec_name, sec_text in sections.items():
                sec_lower = sec_text.lower()
                found = 0
                for kw in all_job_skills:
                    kw_lower = kw.lower().strip()
                    if not kw_lower:
                        continue
                    if (
                        re.search(r"\b" + re.escape(kw_lower) + r"\b", sec_lower)
                        or len(kw_lower.split()) > 1
                        and kw_lower in sec_lower
                    ):
                        found += 1
                tailored_section_scores[sec_name] = round((found / total_skills) * 100, 1)
            gap_report.section_scores = tailored_section_scores
            logger.info(f"Tailored section scores: {tailored_section_scores}")
        except Exception as e:
            logger.warning(f"Failed to compute tailored section scores: {e}")

        # ── Assemble final output ────────────────────────────────────────────
        match_report = MatchReport(
            job_title=job_analysis.job_title,
            company=job_analysis.company,
            overall_match_score=gap_report.match_score,
            cosine_similarity=gap_report.cosine_similarity,
            missing_keywords=gap_report.missing_keywords,
            keyword_frequency=gap_report.keyword_frequency,
            seniority_calibration=gap_report.seniority_calibration,
            ats_checks=ats_resume.ats_checks,
            rewrite_mode=pipeline_config.rewrite_mode.value,
            optimisation_summary=(
                f"Match score: {gap_report.match_score:.1f}% | "
                f"Cosine similarity: {gap_report.cosine_similarity:.4f} | "
                f"Missing keywords: {len(gap_report.missing_keywords)} | "
                f"Weak alignments: {len(gap_report.weak_alignment)}"
            ),
        )

        resume_docx_name = safe_filename(job_analysis.job_title, job_analysis.company, "Resume.docx")
        resume_md_name = safe_filename(job_analysis.job_title, job_analysis.company, "Resume.md")
        report_name = safe_filename(job_analysis.job_title, job_analysis.company, "Match_Report.json")
        tp_name = safe_filename(job_analysis.job_title, job_analysis.company, "Talking_Points.md")

        save_markdown(ats_resume.markdown_content, output_dir / resume_md_name)
        template_pdf_names = []
        for tpl_name in ALL_TEMPLATE_NAMES:
            pdf_name = safe_filename(job_analysis.job_title, job_analysis.company, f"{tpl_name.title()}.pdf")
            generate_resume_pdf(ats_resume.markdown_content, output_dir / pdf_name, template=tpl_name)
            template_pdf_names.append(pdf_name)
        generate_resume_docx(ats_resume.markdown_content, output_dir / resume_docx_name, template=template)
        save_json(match_report.model_dump(), output_dir / report_name)
        save_markdown(format_talking_points(talking_points), output_dir / tp_name)

        emit(6, 6, "Final Assembly", "done", f"{len(talking_points)} talking points generated")

        # ── Post-pipeline enrichment ─────────────────────────────────────────
        cover_letter_md = ""
        resume_quality_data = None
        email_templates_md = ""
        keyword_density_data = None

        try:
            from resume_quality import analyze_resume, extract_bullets_from_markdown

            original_bullets = extract_bullets_from_markdown(resume_text)
            tailored_bullets = extract_bullets_from_markdown(ats_resume.markdown_content)
            quality_before = analyze_resume(original_bullets)
            quality_after = analyze_resume(tailored_bullets)
            resume_quality_data = {
                "before": {
                    "score": quality_before.overall_score,
                    "metrics_pct": quality_before.metrics_percentage,
                    "weak_verbs": quality_before.weak_verbs_used,
                    "filler_words": quality_before.filler_words_found,
                },
                "after": {
                    "score": quality_after.overall_score,
                    "metrics_pct": quality_after.metrics_percentage,
                    "weak_verbs": quality_after.weak_verbs_used,
                    "filler_words": quality_after.filler_words_found,
                },
                "improvement": quality_after.overall_score - quality_before.overall_score,
            }
        except Exception as e:
            logger.warning(f"Resume quality analysis failed: {e}")

        try:
            from email_templates import format_templates_as_markdown, generate_follow_up_templates

            templates = generate_follow_up_templates(
                candidate_name=getattr(resume_data, "name", ""),
                job_title=job_analysis.job_title,
                company=job_analysis.company,
                key_skills=job_analysis.required_skills[:3] if job_analysis.required_skills else [],
            )
            email_templates_md = format_templates_as_markdown(templates)
        except Exception as e:
            logger.warning(f"Email templates generation failed: {e}")

        try:
            from keyword_density import analyze_keyword_density

            kd_report = analyze_keyword_density(
                job_description=job_text,
                original_resume=resume_text,
                tailored_resume=ats_resume.markdown_content,
                required_skills=job_analysis.required_skills,
                preferred_skills=job_analysis.preferred_skills,
                tools=job_analysis.tools,
            )
            keyword_density_data = {
                "total": kd_report.total_jd_keywords,
                "matched_before": kd_report.matched_before,
                "matched_after": kd_report.matched_after,
                "improvement": kd_report.improvement,
                "keywords": [
                    {
                        "keyword": k.keyword,
                        "jd": k.jd_count,
                        "before": k.resume_before_count,
                        "after": k.resume_after_count,
                        "status": k.status,
                    }
                    for k in kd_report.keywords[:30]
                ],
            }
        except Exception as e:
            logger.warning(f"Keyword density analysis failed: {e}")

        try:
            from agents.cover_letter import CoverLetterAgent

            cover_agent = CoverLetterAgent(pipeline_config)
            cover_result = cover_agent.run(
                {
                    "job_analysis": job_analysis,
                    "resume_md": ats_resume.markdown_content,
                    "candidate_name": getattr(resume_data, "name", ""),
                }
            )
            cover_letter_md = cover_result.cover_letter_md
        except Exception as e:
            logger.warning(f"Cover letter generation failed (non-critical): {e}")
            cover_letter_md = None

        if cover_letter_md:
            try:
                cl_pdf_name = safe_filename(job_analysis.job_title, job_analysis.company, "Cover_Letter.pdf")
                generate_resume_pdf(cover_letter_md, output_dir / cl_pdf_name, template=template)
            except Exception as e:
                logger.warning(f"Cover letter PDF generation failed: {e}")
                cl_pdf_name = None
            try:
                cl_docx_name = safe_filename(
                    job_analysis.job_title, job_analysis.company, "Cover_Letter.docx"
                )
                generate_resume_docx(cover_letter_md, output_dir / cl_docx_name, template=template)
            except Exception as e:
                logger.warning(f"Cover letter DOCX generation failed: {e}")
                cl_docx_name = None
        else:
            cl_pdf_name = None
            cl_docx_name = None

        total_elapsed = time.time() - pipeline_start
        logger.info(f"Total pipeline completed in {total_elapsed:.1f}s")

        job_stats = pipeline_analytics.complete_job(job_id)
        if job_stats:
            logger.info(
                f"Analytics: {job_stats.get('total_tokens', 0)} tokens, "
                f"{job_stats.get('api_calls', 0)} API calls, "
                f"${job_stats.get('estimated_cost_usd', 0):.4f} est. cost"
            )

        _te("tailor.job.completed", metadata={
            "duration_s": round(total_elapsed, 1),
            "model": model,
            "total_tokens": job_stats.get("total_tokens", 0) if job_stats else 0,
            "api_calls": job_stats.get("api_calls", 0) if job_stats else 0,
            "retries": job_stats.get("retries", 0) if job_stats else 0,
            "estimated_cost_usd": job_stats.get("estimated_cost_usd", 0) if job_stats else 0,
            "tailored_match_score": tailored_match_score,
            "original_match_score": gap_report.match_score,
            "queue_wait_ms": queue_wait_ms,
        })

        # ── Persist to DB and upload to R2 ───────────────────────────────────
        files_list = template_pdf_names + [resume_docx_name, resume_md_name, report_name, tp_name]
        if cl_pdf_name:
            files_list.append(cl_pdf_name)
        if cl_docx_name:
            files_list.append(cl_docx_name)

        with flask_app.app_context():
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
                db_job.recruiter_resume_md = ats_resume.markdown_content
                db_job.talking_points_md = format_talking_points(talking_points)
                db_job.cover_letter_md = cover_letter_md or None
                db_job.section_scores = (
                    gap_report.section_scores if hasattr(gap_report, "section_scores") else None
                )
                db_job.resume_quality_json = resume_quality_data
                db_job.email_templates_md = email_templates_md or None
                db_job.keyword_density_json = keyword_density_data
                db_job.job_description_snippet = job_text[:500] if job_text else None
                db_job.completed_at = datetime.now(timezone.utc)
                db_job.duration_seconds = total_elapsed

                for filename in files_list:
                    file_path = output_dir / filename
                    if file_path.exists():
                        r2_key = ""
                        if r2_storage.is_configured:
                            try:
                                r2_key = r2_storage.upload_file(job_id, filename, file_path=file_path)
                            except Exception as upload_err:
                                logger.error(f"R2 upload failed for {filename}: {upload_err}")
                        # Always create JobFile record (for filename tracking + DB fallback)
                        job_file = JobFile(
                            job_id=job_id,
                            filename=filename,
                            r2_key=r2_key,
                            content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
                            size_bytes=file_path.stat().st_size,
                        )
                        db.session.add(job_file)

                db.session.commit()
                logger.info(f"Job {job_id} persisted to database")

        def _cleanup_output(path: Path, delay: int = 300) -> None:
            time.sleep(delay)
            shutil.rmtree(str(path), ignore_errors=True)

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
                "recruiter_resume_md": ats_resume.markdown_content,
                "original_resume_text": resume_text,
                "talking_points_md": format_talking_points(talking_points),
                "cover_letter_md": cover_letter_md,
                "section_scores": gap_report.section_scores if hasattr(gap_report, "section_scores") else {},
                "resume_quality": resume_quality_data,
                "email_templates_md": email_templates_md,
                "keyword_density": keyword_density_data,
                "files": files_list,
            }
        progress_queue.put({"status": "complete"})

        gc.collect()

        def _delayed_result_cleanup(jid: str, delay: int = 120) -> None:
            time.sleep(delay)
            with jobs_lock:
                if jid in jobs and jobs[jid].get("status") == "complete":
                    jobs[jid].pop("result", None)

        threading.Thread(target=_delayed_result_cleanup, args=(job_id,), daemon=True).start()

    except Exception as e:
        logging.getLogger("cvtailro.pipeline").exception("Pipeline failed")
        error_msg = str(e)
        pipeline_analytics.complete_job(job_id)
        _te("tailor.job.failed", metadata={"model": model, "error": error_msg[:500], "queue_wait_ms": queue_wait_ms})
        import traceback

        with pipeline_errors_lock:
            pipeline_errors.append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "model": model,
                    "error": error_msg,
                    "traceback": traceback.format_exc()[-500:],
                }
            )
            if len(pipeline_errors) > 50:
                pipeline_errors.pop(0)

        try:
            with flask_app.app_context():
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
        try:
            if log_cleanup is not None:
                log_cleanup()
        except Exception:
            pass
        pipeline_semaphore.release()
