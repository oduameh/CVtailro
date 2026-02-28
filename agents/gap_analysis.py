"""Agent 3: Gap Analysis — Pure Python Implementation.

Compares structured job analysis against structured resume data
to identify gaps, misalignments, and optimisation priorities.
All computation is done in Python — no LLM call required.
This eliminates ~90 seconds from the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config import AppConfig
from models import (
    GapReport,
    JobAnalysis,
    KeywordFrequency,
    PrioritisedSkill,
    ResumeData,
    SkillPriority,
)
from similarity import keyword_frequency_analysis, resume_job_similarity

logger = logging.getLogger(__name__)


class GapAnalysisAgent:
    """Analyses gaps between job requirements and resume content.

    This is a pure Python agent — no LLM call. It computes:
    - Cosine similarity via TF-IDF
    - Missing and weak keywords via set operations
    - Keyword frequency via regex counting
    - Match score derived from cosine similarity and keyword overlap
    - Prioritised skill recommendations
    """

    AGENT_NAME = "Gap Analysis Agent"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, input_data: Any, **kwargs: Any) -> GapReport:
        """Run the gap analysis entirely in Python."""
        job_analysis: JobAnalysis = input_data["job_analysis"]
        resume_data: ResumeData = input_data["resume_data"]

        logger.info(f"[{self.AGENT_NAME}] Computing gap analysis (pure Python)...")

        job_text = job_analysis.raw_text_for_similarity
        resume_text = resume_data.raw_text_for_similarity

        # ── Cosine similarity ──
        cos_sim = 0.0
        if job_text and resume_text:
            cos_sim = resume_job_similarity(resume_text, job_text)
        logger.info(f"[{self.AGENT_NAME}] Cosine similarity: {cos_sim:.4f}")

        # ── Keyword analysis ──
        resume_skills_lower = {s.lower() for s in resume_data.global_skills}
        resume_text_lower = resume_text.lower() if resume_text else ""

        required = job_analysis.required_skills
        preferred = job_analysis.preferred_skills
        tools = job_analysis.tools

        def skill_in_resume(skill: str) -> bool:
            sl = skill.strip().lower()
            if not sl:
                return True  # treat empty strings as "present" to skip them
            if sl in resume_skills_lower:
                return True
            if resume_text_lower and re.search(
                r"\b" + re.escape(sl) + r"\b", resume_text_lower
            ):
                return True
            return False

        missing_keywords = [
            s for s in required + preferred + tools if not skill_in_resume(s)
        ]

        # Skills present but underrepresented
        all_job_keywords = [
            kw for kw in set(required + preferred + tools + job_analysis.domain_keywords)
            if kw.strip()
        ]
        weak_alignment: list[str] = []
        overrepresented: list[str] = []
        keyword_frequency: list[KeywordFrequency] = []

        if job_text and resume_text:
            freq_data = keyword_frequency_analysis(
                all_job_keywords, resume_text, job_text
            )
            keyword_frequency = [KeywordFrequency(**item) for item in freq_data]

            for kf in keyword_frequency:
                if kf.resume_count > 0 and kf.gap > 0:
                    weak_alignment.append(kf.keyword)
                elif kf.resume_count > kf.job_count and kf.job_count == 0:
                    overrepresented.append(kf.keyword)

        # ── Optimisation priority ──
        priority_list: list[PrioritisedSkill] = []
        for skill in missing_keywords:
            if skill in required:
                prio = SkillPriority.CRITICAL
                ctx = "Required skill not found in resume"
            elif skill in tools:
                prio = SkillPriority.HIGH
                ctx = "Required tool not found in resume"
            elif skill in preferred:
                prio = SkillPriority.MEDIUM
                ctx = "Preferred skill not found in resume"
            else:
                prio = SkillPriority.LOW
                ctx = "Domain keyword not found in resume"
            priority_list.append(
                PrioritisedSkill(skill=skill, priority=prio, context=ctx)
            )

        priority_order = {
            SkillPriority.CRITICAL: 0,
            SkillPriority.HIGH: 1,
            SkillPriority.MEDIUM: 2,
            SkillPriority.LOW: 3,
        }
        priority_list.sort(key=lambda p: priority_order.get(p.priority, 3))

        # ── Match score ──
        total_job_skills = len(set(required + preferred + tools))
        keyword_coverage = 1.0 - (
            len(missing_keywords) / max(total_job_skills, 1)
        )
        match_score = round((cos_sim * 40) + (keyword_coverage * 40) + 20, 1)
        match_score = max(0.0, min(100.0, match_score))

        # ── Seniority calibration ──
        seniority_map = {
            "JUNIOR": 2, "MID": 4, "SENIOR": 7, "STAFF": 10,
            "PRINCIPAL": 12, "DIRECTOR": 15, "VP": 18, "C_LEVEL": 20,
        }
        expected_years = seniority_map.get(
            job_analysis.inferred_seniority.value, 5
        )
        actual_years = resume_data.total_years_estimate
        if actual_years >= expected_years:
            seniority_cal = (
                f"Good fit — {actual_years:.0f} years experience meets "
                f"{job_analysis.inferred_seniority.value} level expectations"
            )
        elif actual_years >= expected_years * 0.7:
            seniority_cal = (
                f"Slight gap — {actual_years:.0f} years vs ~{expected_years} "
                f"expected for {job_analysis.inferred_seniority.value} level"
            )
        else:
            seniority_cal = (
                f"Significant gap — {actual_years:.0f} years vs "
                f"~{expected_years} expected for "
                f"{job_analysis.inferred_seniority.value} level"
            )

        gap_report = GapReport(
            match_score=match_score,
            cosine_similarity=round(cos_sim, 4),
            missing_keywords=missing_keywords,
            weak_alignment=weak_alignment[:15],
            overrepresented_skills=overrepresented[:10],
            keyword_frequency=keyword_frequency,
            optimisation_priority=priority_list,
            seniority_calibration=seniority_cal,
            recommended_skill_order=[p.skill for p in priority_list],
        )

        logger.info(
            f"[{self.AGENT_NAME}] Completed: match={match_score:.0f}%, "
            f"missing={len(missing_keywords)}, weak={len(weak_alignment)}"
        )

        return gap_report
