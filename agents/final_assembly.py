"""Agent 7: Final Assembly Agent.

Assembles all output artifacts: ATS resume, recruiter resume,
match report, and interview talking points. Uses one LLM call
for talking points generation; the rest is deterministic assembly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from base_agent import BaseAgent
from models import (
    ATSResume,
    FinalOutput,
    GapReport,
    JobAnalysis,
    MatchReport,
    RecruiterResume,
    TalkingPoint,
)

logger = logging.getLogger(__name__)


class FinalAssemblyAgent(BaseAgent["FinalOutput"]):
    """Assembles final output artifacts and generates talking points."""

    PROMPT_FILE = "final_assembly.txt"
    OUTPUT_MODEL = FinalOutput
    AGENT_NAME = "Final Assembly Agent"
    AGENT_MAX_TOKENS = 4096

    def prepare_user_message(self, input_data: Any) -> str:
        """Format prior outputs for talking points generation.

        Accepts either 'ats_resume' (ATSResume) or 'optimised_bullets'
        (OptimisedBullets) + 'resume_data' (ResumeData) as the resume
        source. This allows talking points to be generated in parallel
        with the ATS/Recruiter optimisers.
        """
        gap: GapReport = input_data["gap_report"]
        job: JobAnalysis = input_data["job_analysis"]

        # Use ATS resume if available, otherwise fall back to optimised bullets
        if "ats_resume" in input_data and input_data["ats_resume"] is not None:
            resume_content = input_data["ats_resume"].markdown_content
        else:
            # Build resume summary from optimised bullets
            optimised = input_data.get("optimised_bullets")
            resume_data = input_data.get("resume_data")
            lines = []
            if optimised and optimised.summary_rewrite:
                lines.append(f"SUMMARY: {optimised.summary_rewrite}")
            if resume_data:
                for role in resume_data.roles:
                    lines.append(f"\n{role.title} @ {role.company}")
                    for b in role.bullets:
                        lines.append(f"- {b.original_text}")
            resume_content = "\n".join(lines) if lines else "N/A"

        return (
            "TARGET JOB:\n"
            f"Title: {job.job_title}\n"
            f"Company: {job.company}\n"
            f"Key Requirements: {json.dumps(job.required_skills)}\n"
            f"Responsibilities: {json.dumps(job.responsibilities[:8])}\n\n"
            "GAP ANALYSIS:\n"
            f"Match Score: {gap.match_score}%\n"
            f"Missing Keywords: {json.dumps(gap.missing_keywords)}\n"
            f"Weak Alignment: {json.dumps(gap.weak_alignment)}\n"
            f"Seniority Calibration: {gap.seniority_calibration}\n\n"
            "RESUME CONTENT:\n"
            f"{resume_content}\n"
        )

    def run(self, input_data: Any, **prompt_vars: Any) -> FinalOutput:
        """Override run to combine LLM-generated talking points with
        deterministic assembly of other outputs.
        """
        ats: ATSResume = input_data["ats_resume"]
        recruiter: RecruiterResume = input_data["recruiter_resume"]
        gap: GapReport = input_data["gap_report"]
        job: JobAnalysis = input_data["job_analysis"]

        # Generate talking points via LLM
        talking_points = self._generate_talking_points(input_data)

        # Assemble match report deterministically
        match_report = MatchReport(
            job_title=job.job_title,
            company=job.company,
            overall_match_score=gap.match_score,
            cosine_similarity=gap.cosine_similarity,
            missing_keywords=gap.missing_keywords,
            keyword_frequency=gap.keyword_frequency,
            seniority_calibration=gap.seniority_calibration,
            ats_checks=ats.ats_checks,
            rewrite_mode=self.config.rewrite_mode.value,
            optimisation_summary=(
                f"Match score: {gap.match_score:.1f}% | "
                f"Cosine similarity: {gap.cosine_similarity:.4f} | "
                f"Missing keywords: {len(gap.missing_keywords)} | "
                f"Weak alignments: {len(gap.weak_alignment)}"
            ),
        )

        return FinalOutput(
            ats_resume_md=ats.markdown_content,
            recruiter_resume_md=recruiter.markdown_content,
            match_report=match_report,
            talking_points=talking_points,
        )

    def _generate_talking_points(
        self, input_data: Any
    ) -> list[TalkingPoint]:
        """Use the OpenRouter API to generate interview talking points."""
        system = self.format_system_prompt()
        user_message = self.prepare_user_message(input_data)

        logger.info(f"[{self.AGENT_NAME}] Generating interview talking points...")

        try:
            raw_text = self._call_llm_api(system, user_message)
            parsed_json = self._extract_json(raw_text)

            # Handle both {"talking_points": [...]} and bare [...]
            if isinstance(parsed_json, dict) and "talking_points" in parsed_json:
                points_data = parsed_json["talking_points"]
            elif isinstance(parsed_json, list):
                points_data = parsed_json
            else:
                points_data = parsed_json.get("talking_points", [])

            return [TalkingPoint.model_validate(p) for p in points_data]

        except Exception as e:
            logger.warning(
                f"[{self.AGENT_NAME}] Failed to generate talking points: {e}. "
                f"Returning empty list."
            )
            return []
