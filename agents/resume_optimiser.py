"""Agent 5: Unified Resume Optimisation Agent.

Produces a single resume that is both ATS-compliant and recruiter-compelling.
Replaces the previous separate ATS Optimiser (Agent 5) and Recruiter Optimiser
(Agent 6) with a unified approach.

Ensures standard headings, no tables/icons, keyword density, job-title
alignment, professional narrative, and impact-driven language.
"""

from __future__ import annotations

import json
from typing import Any

from base_agent import BaseAgent
from models import ATSResume, GapReport, JobAnalysis, OptimisedBullets, ResumeData


class ResumeOptimiserAgent(BaseAgent["ATSResume"]):
    """Assembles a unified resume optimised for both ATS and recruiters."""

    PROMPT_FILE = "resume_optimiser.txt"
    OUTPUT_MODEL = ATSResume
    AGENT_NAME = "Resume Optimiser Agent"
    AGENT_MAX_TOKENS = 8192

    def prepare_user_message(self, input_data: Any) -> str:
        """Format optimised bullets, job analysis, gap analysis, and resume data.

        Args:
            input_data: Dict with 'optimised_bullets' (OptimisedBullets),
                       'job_analysis' (JobAnalysis),
                       'gap_report' (GapReport), and
                       'resume_data' (ResumeData) keys.
        """
        optimised: OptimisedBullets = input_data["optimised_bullets"]
        job: JobAnalysis = input_data["job_analysis"]
        gap: GapReport = input_data["gap_report"]
        resume: ResumeData = input_data["resume_data"]

        # Build the optimised bullets mapped to their roles
        bullets_by_role: dict[int, list[dict[str, str]]] = {}
        for b in optimised.bullets:
            if b.role_index not in bullets_by_role:
                bullets_by_role[b.role_index] = []
            bullets_by_role[b.role_index].append(
                {
                    "bullet_index": b.bullet_index,
                    "text": b.optimised_text,
                }
            )

        # Resume structure with optimised bullets
        roles_data = []
        for idx, role in enumerate(resume.roles):
            role_bullets = bullets_by_role.get(idx, [])
            if not role_bullets:
                # Use original bullets if no optimised version
                role_bullets = [
                    {"bullet_index": bi, "text": b.original_text}
                    for bi, b in enumerate(role.bullets)
                ]

            roles_data.append(
                {
                    "title": role.title,
                    "company": role.company,
                    "location": role.location,
                    "start_date": role.start_date,
                    "end_date": role.end_date,
                    "duration_months": role.duration_months,
                    "bullets": role_bullets,
                }
            )

        return (
            "CANDIDATE INFORMATION:\n"
            f"Name: {resume.name}\n"
            f"Contact: {json.dumps(resume.contact_info)}\n"
            f"Total Experience: {resume.total_years_estimate} years\n\n"
            "PROFESSIONAL SUMMARY (optimised):\n"
            f"{optimised.summary_rewrite or resume.summary}\n\n"
            "EXPERIENCE (with optimised bullets):\n"
            f"{json.dumps(roles_data, indent=2)}\n\n"
            "EDUCATION:\n"
            f"{json.dumps([e.model_dump() for e in resume.education], indent=2)}\n\n"
            "CERTIFICATIONS:\n"
            f"{json.dumps([c.model_dump() for c in resume.certifications], indent=2)}\n\n"
            "SKILLS (in recommended priority order):\n"
            f"{json.dumps(resume.global_skills)}\n\n"
            "TARGET JOB:\n"
            f"Title: {job.job_title}\n"
            f"Company: {job.company}\n"
            f"Seniority: {job.inferred_seniority.value}\n"
            f"Required Skills: {json.dumps(job.required_skills)}\n"
            f"Preferred Skills: {json.dumps(job.preferred_skills)}\n"
            f"Tools & Technologies: {json.dumps(job.tools)}\n"
            f"Domain Keywords: {json.dumps(job.domain_keywords)}\n\n"
            "GAP ANALYSIS — CRITICAL FOR KEYWORD INTEGRATION:\n"
            f"Match Score: {gap.match_score}%\n"
            f"Seniority Calibration: {gap.seniority_calibration}\n"
            f"MISSING KEYWORDS (MUST integrate ALL of these): {json.dumps(gap.missing_keywords)}\n"
            f"WEAK ALIGNMENT (strengthen these): {json.dumps(gap.weak_alignment)}\n"
        )
