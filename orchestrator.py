#!/usr/bin/env python3
"""
CVtailro: Multi-Agent Resume Tailoring System

A production-grade, multi-agent system that analyses job descriptions,
compares them against a master resume, and produces ATS-optimised and
recruiter-optimised resume versions (both PDF and markdown) with match
reports and interview talking points.

Powered by the OpenRouter API — supports 100+ LLM models.

Usage:
    python orchestrator.py --job job.txt --resume resume.pdf --api-key sk-or-v1-...
    python orchestrator.py --job job.txt --resume resume.pdf --mode aggressive --model openai/gpt-4o
    python orchestrator.py --job job.txt --resume master.md --stage bullet_optimiser --output-dir output/20260227_143000/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from agents.ats_optimiser import ATSOptimiserAgent
from agents.bullet_optimiser import BulletOptimiserAgent
from agents.final_assembly import FinalAssemblyAgent
from agents.gap_analysis import GapAnalysisAgent
from agents.job_intelligence import JobIntelligenceAgent
from agents.recruiter_optimiser import RecruiterOptimiserAgent
from agents.resume_parser import ResumeParserAgent
from config import AppConfig, DEFAULT_MODEL
from models import (
    ATSResume,
    GapReport,
    JobAnalysis,
    OptimisedBullets,
    RecruiterResume,
    ResumeData,
    RewriteMode,
)
from pdf_generator import generate_resume_pdf
from utils import (
    create_output_dir,
    load_file,
    load_resume,
    save_json,
    save_markdown,
    setup_logging,
)

STAGE_ORDER = [
    "job_intelligence",
    "resume_parser",
    "gap_analysis",
    "bullet_optimiser",
    "ats_optimiser",
    "recruiter_optimiser",
    "final_assembly",
]

# Maps stage names to their artifact filenames and Pydantic models
STAGE_ARTIFACTS = {
    "job_intelligence": ("01_job_analysis.json", JobAnalysis),
    "resume_parser": ("02_resume_data.json", ResumeData),
    "gap_analysis": ("03_gap_report.json", GapReport),
    "bullet_optimiser": ("04_optimised_bullets.json", OptimisedBullets),
    "ats_optimiser": ("05_ats_resume.json", ATSResume),
    "recruiter_optimiser": ("06_recruiter_resume.json", RecruiterResume),
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="CVtailro: AI-powered multi-agent resume tailoring system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python orchestrator.py --job job.txt --resume resume.pdf\n"
            "  python orchestrator.py --job job.txt --resume resume.pdf --mode aggressive\n"
            "  python orchestrator.py --job job.txt --resume master.md "
            "--stage bullet_optimiser --output-dir output/20260227_143000/\n"
        ),
    )
    parser.add_argument(
        "--job",
        required=True,
        help="Path to job description file (.txt or .md)",
    )
    parser.add_argument(
        "--resume",
        required=True,
        help="Path to master resume file (.pdf, .md, or .txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Rewriting intensity (default: conservative)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        default=None,
        help="Re-run from a specific stage (requires --output-dir with prior run artifacts)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated timestamped dir under output/)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    return parser


def load_artifact(output_dir: Path, stage_name: str) -> object:
    """Load a previously saved artifact from the output directory.

    Args:
        output_dir: Path to the output directory.
        stage_name: Stage name (e.g., 'job_intelligence').

    Returns:
        Validated Pydantic model instance.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    filename, model_cls = STAGE_ARTIFACTS[stage_name]
    artifact_path = output_dir / filename
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}. "
            f"Cannot re-run from stage '{stage_name}' without prior artifacts."
        )
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    return model_cls.model_validate(data)


def format_talking_points(talking_points: list) -> str:
    """Format talking points into a readable markdown document."""
    lines = ["# Interview Talking Points\n"]

    for i, tp in enumerate(talking_points, 1):
        lines.append(f"## {i}. {tp.topic}\n")
        if tp.source_experience:
            lines.append(f"*Based on: {tp.source_experience}*\n")
        for point in tp.bullet_points:
            lines.append(f"- {point}")
        lines.append("")

    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full agent pipeline or re-run from a specific stage."""
    config = AppConfig(
        rewrite_mode=RewriteMode(args.mode),
        output_dir=args.output_dir,
        verbose=args.verbose,
        api_key=args.api_key,
        model=args.model,
    )

    config.validate_api_config()

    output_dir = create_output_dir(config.output_dir)
    setup_logging(verbose=config.verbose, log_file=output_dir / "pipeline.log")
    logger = logging.getLogger("orchestrator")

    logger.info("=" * 60)
    logger.info("CVtailro: Multi-Agent Resume Tailoring System")
    logger.info(f"Model: {config.model} via OpenRouter")
    logger.info(f"Rewrite mode: {config.rewrite_mode.value}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Load raw inputs
    job_text = load_file(args.job)
    resume_text = load_resume(args.resume)  # Handles both PDF and text

    logger.info(f"Resume loaded: {Path(args.resume).name} ({len(resume_text)} chars)")
    logger.info(f"Job description loaded: {Path(args.job).name} ({len(job_text)} chars)")

    # Determine which stages to run
    if args.stage:
        start_idx = STAGE_ORDER.index(args.stage)
        logger.info(f"Re-running from stage: {args.stage}")
    else:
        start_idx = 0

    pipeline_start = time.time()

    # ── Load prior artifacts if re-running from a later stage ──
    job_analysis: JobAnalysis | None = None
    resume_data: ResumeData | None = None
    gap_report: GapReport | None = None
    optimised_bullets: OptimisedBullets | None = None
    ats_resume: ATSResume | None = None
    recruiter_resume: RecruiterResume | None = None

    if start_idx > 0:
        logger.info("Loading prior artifacts...")
        if start_idx > 0:
            job_analysis = load_artifact(output_dir, "job_intelligence")
        if start_idx > 1:
            resume_data = load_artifact(output_dir, "resume_parser")
        if start_idx > 2:
            gap_report = load_artifact(output_dir, "gap_analysis")
        if start_idx > 3:
            optimised_bullets = load_artifact(output_dir, "bullet_optimiser")
        if start_idx > 4:
            ats_resume = load_artifact(output_dir, "ats_optimiser")
        if start_idx > 5:
            recruiter_resume = load_artifact(output_dir, "recruiter_optimiser")

    # ── STAGE 1: Job Intelligence ─────────────────────────────
    if start_idx <= 0:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 1/7: Job Intelligence Agent")
        logger.info("=" * 60)
        stage_start = time.time()

        agent1 = JobIntelligenceAgent(config)
        job_analysis = agent1.run(job_text)
        save_json(job_analysis.model_dump(), output_dir / "01_job_analysis.json")

        logger.info(
            f"  -> Extracted {len(job_analysis.required_skills)} required skills, "
            f"{len(job_analysis.preferred_skills)} preferred skills, "
            f"{len(job_analysis.tools)} tools"
        )
        logger.info(f"  -> Inferred seniority: {job_analysis.inferred_seniority.value}")
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 2: Resume Parser ────────────────────────────────
    if start_idx <= 1:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 2/7: Resume Parser Agent")
        logger.info("=" * 60)
        stage_start = time.time()

        agent2 = ResumeParserAgent(config)
        resume_data = agent2.run(resume_text)
        save_json(resume_data.model_dump(), output_dir / "02_resume_data.json")

        total_bullets = sum(len(r.bullets) for r in resume_data.roles)
        logger.info(
            f"  -> Parsed {len(resume_data.roles)} roles, "
            f"{total_bullets} bullets, "
            f"{len(resume_data.global_skills)} skills"
        )
        logger.info(
            f"  -> Total experience: {resume_data.total_years_estimate} years"
        )
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 3: Gap Analysis ─────────────────────────────────
    if start_idx <= 2:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 3/7: Gap Analysis Agent")
        logger.info("=" * 60)
        stage_start = time.time()

        agent3 = GapAnalysisAgent(config)
        gap_report = agent3.run(
            {"job_analysis": job_analysis, "resume_data": resume_data}
        )
        save_json(gap_report.model_dump(), output_dir / "03_gap_report.json")

        logger.info(f"  -> Match score: {gap_report.match_score:.1f}%")
        logger.info(f"  -> Cosine similarity: {gap_report.cosine_similarity:.4f}")
        logger.info(
            f"  -> Missing keywords: {len(gap_report.missing_keywords)}"
        )
        logger.info(
            f"  -> Weak alignments: {len(gap_report.weak_alignment)}"
        )
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 4: Bullet Optimiser ─────────────────────────────
    if start_idx <= 3:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 4/7: Bullet Optimiser Agent")
        logger.info(f"  Mode: {config.rewrite_mode.value}")
        logger.info("=" * 60)
        stage_start = time.time()

        agent4 = BulletOptimiserAgent(config)
        optimised_bullets = agent4.run(
            {"resume_data": resume_data, "gap_report": gap_report},
            rewrite_mode=config.rewrite_mode.value,
            job_title=job_analysis.job_title,
            company=job_analysis.company or "the target company",
            job_responsibilities="; ".join(job_analysis.responsibilities[:5]),
            required_skills=", ".join(job_analysis.required_skills[:10]),
        )
        save_json(
            optimised_bullets.model_dump(),
            output_dir / "04_optimised_bullets.json",
        )

        fabrication_count = sum(
            1 for b in optimised_bullets.bullets if b.fabrication_flag
        )
        logger.info(
            f"  -> Optimised {len(optimised_bullets.bullets)} bullets"
        )
        if fabrication_count:
            logger.warning(
                f"  -> {fabrication_count} bullets flagged for potential fabrication"
            )
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 5: ATS Optimiser ────────────────────────────────
    if start_idx <= 4:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 5/7: ATS Optimiser Agent")
        logger.info("=" * 60)
        stage_start = time.time()

        agent5 = ATSOptimiserAgent(config)
        ats_resume = agent5.run(
            {
                "optimised_bullets": optimised_bullets,
                "job_analysis": job_analysis,
                "resume_data": resume_data,
            }
        )
        save_json(ats_resume.model_dump(), output_dir / "05_ats_resume.json")

        checks_passed = sum(1 for c in ats_resume.ats_checks if c.passed)
        logger.info(
            f"  -> ATS checks: {checks_passed}/{len(ats_resume.ats_checks)} passed"
        )
        logger.info(
            f"  -> Job title aligned: {ats_resume.job_title_aligned}"
        )
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 6: Recruiter Optimiser ──────────────────────────
    if start_idx <= 5:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 6/7: Recruiter Optimiser Agent")
        logger.info("=" * 60)
        stage_start = time.time()

        agent6 = RecruiterOptimiserAgent(config)
        recruiter_resume = agent6.run(
            {
                "optimised_bullets": optimised_bullets,
                "job_analysis": job_analysis,
                "gap_report": gap_report,
                "resume_data": resume_data,
            }
        )
        save_json(
            recruiter_resume.model_dump(),
            output_dir / "06_recruiter_resume.json",
        )

        logger.info(
            f"  -> Narrative improvements: {len(recruiter_resume.narrative_improvements)}"
        )
        logger.info(
            f"  -> Leadership signals: {len(recruiter_resume.leadership_signals_added)}"
        )
        logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── STAGE 7: Final Assembly ───────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 7/7: Final Assembly Agent")
    logger.info("=" * 60)
    stage_start = time.time()

    agent7 = FinalAssemblyAgent(config)
    final_output = agent7.run(
        {
            "ats_resume": ats_resume,
            "recruiter_resume": recruiter_resume,
            "gap_report": gap_report,
            "job_analysis": job_analysis,
        }
    )

    # ── Write final artifacts ─────────────────────────────────

    # Markdown versions
    save_markdown(
        final_output.ats_resume_md, output_dir / "tailored_resume_ats.md"
    )
    save_markdown(
        final_output.recruiter_resume_md,
        output_dir / "tailored_resume_recruiter.md",
    )

    # PDF versions
    generate_resume_pdf(
        final_output.ats_resume_md,
        output_dir / "tailored_resume_ats.pdf",
    )
    generate_resume_pdf(
        final_output.recruiter_resume_md,
        output_dir / "tailored_resume_recruiter.pdf",
    )

    # Reports
    save_json(
        final_output.match_report.model_dump(),
        output_dir / "match_report.json",
    )
    save_markdown(
        format_talking_points(final_output.talking_points),
        output_dir / "interview_talking_points.md",
    )

    total_time = time.time() - pipeline_start
    logger.info(f"  -> Generated {len(final_output.talking_points)} talking points")
    logger.info(f"  -> Completed in {time.time() - stage_start:.1f}s")

    # ── Summary ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(
        f"Overall match score: {final_output.match_report.overall_match_score:.1f}%"
    )
    logger.info(
        f"Cosine similarity: {final_output.match_report.cosine_similarity:.4f}"
    )
    logger.info(f"Rewrite mode: {final_output.match_report.rewrite_mode}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  {output_dir / 'tailored_resume_ats.pdf'}")
    logger.info(f"  {output_dir / 'tailored_resume_recruiter.pdf'}")
    logger.info(f"  {output_dir / 'tailored_resume_ats.md'}")
    logger.info(f"  {output_dir / 'tailored_resume_recruiter.md'}")
    logger.info(f"  {output_dir / 'match_report.json'}")
    logger.info(f"  {output_dir / 'interview_talking_points.md'}")
    logger.info(f"  {output_dir / 'pipeline.log'}")
    logger.info("")
    logger.info(f"Intermediate artifacts saved in: {output_dir}")


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Resolve API key: CLI arg > env var
    if not args.api_key:
        args.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.api_key:
        print(
            "Error: --api-key is required (or set OPENROUTER_API_KEY env var).\n"
            "Get your key at https://openrouter.ai/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate input files exist
    if not Path(args.job).exists():
        print(f"Error: Job description file not found: {args.job}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.resume).exists():
        print(f"Error: Resume file not found: {args.resume}", file=sys.stderr)
        sys.exit(1)

    # Validate stage re-run requirements
    if args.stage and not args.output_dir:
        print(
            "Error: --output-dir is required when using --stage "
            "(need prior artifacts to re-run from a specific stage).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        run_pipeline(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.getLogger("orchestrator").exception("Pipeline failed")
        print(f"Pipeline error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
