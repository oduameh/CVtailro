"""
Pydantic models defining the typed contracts between all agents.

This is the single source of truth for every data structure in the CVtailro
pipeline. Each agent's input/output is defined here to ensure strict type
safety and validation across the entire system.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────────


class RewriteMode(str, Enum):
    """Controls how aggressively bullets are rewritten."""

    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class SeniorityLevel(str, Enum):
    """Inferred seniority from a job description or resume."""

    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


class SkillPriority(str, Enum):
    """Priority ranking for a skill relative to a job description."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─── Agent 1: Job Intelligence ──────────────────────────────────────────────────


class PrioritisedSkill(BaseModel):
    """A skill with its priority ranking and context."""

    skill: str
    priority: SkillPriority
    context: str = Field(
        default="",
        description="Where/how this skill appears in the job description",
    )


class JobAnalysis(BaseModel):
    """Structured output from the Job Intelligence Agent."""

    job_title: str
    company: str = ""
    department: str = ""
    required_skills: list[str]
    preferred_skills: list[str]
    tools: list[str]
    seniority_signals: list[str]
    inferred_seniority: SeniorityLevel
    soft_skills: list[str]
    domain_keywords: list[str]
    responsibilities: list[str]
    inferred_priority_skills: list[PrioritisedSkill]
    raw_text_for_similarity: str = Field(
        default="",
        description="Cleaned job text for cosine similarity computation",
    )


# ─── Agent 2: Resume Parser ─────────────────────────────────────────────────────


class BulletPoint(BaseModel):
    """A single resume bullet point with detected metadata."""

    original_text: str
    skills_detected: list[str] = []
    metrics_detected: list[str] = []
    action_verb: str = ""


class Role(BaseModel):
    """A single work experience entry."""

    title: str
    company: str
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_months: int = 0
    bullets: list[BulletPoint]
    skills_detected: list[str] = []


class Education(BaseModel):
    """An education entry."""

    institution: str
    degree: str
    field: str = ""
    start_date: str = ""
    end_date: str = ""
    highlights: list[str] = []


class Certification(BaseModel):
    """A professional certification."""

    name: str
    issuer: str = ""
    date: str = ""


class ResumeData(BaseModel):
    """Structured output from the Resume Parsing Agent."""

    name: str
    contact_info: dict[str, str] = {}
    summary: str = ""
    roles: list[Role]
    education: list[Education] = []
    certifications: list[Certification] = []
    global_skills: list[str]
    total_years_estimate: float
    raw_text_for_similarity: str = Field(
        default="",
        description="Cleaned resume text for cosine similarity computation",
    )


# ─── Agent 3: Gap Analysis ──────────────────────────────────────────────────────


class KeywordFrequency(BaseModel):
    """Keyword occurrence counts in job vs resume."""

    keyword: str
    job_count: int
    resume_count: int
    gap: int = Field(
        default=0,
        description="job_count - resume_count; positive = underrepresented in resume",
    )


class GapReport(BaseModel):
    """Structured output from the Gap Analysis Agent."""

    match_score: float = Field(
        ge=0.0, le=100.0, description="Overall match percentage (0-100)"
    )
    cosine_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="TF-IDF cosine similarity score"
    )
    missing_keywords: list[str]
    weak_alignment: list[str] = Field(
        default_factory=list,
        description="Skills present in resume but underemphasized",
    )
    overrepresented_skills: list[str] = Field(
        default_factory=list,
        description="Skills emphasized in resume more than job warrants",
    )
    keyword_frequency: list[KeywordFrequency] = []
    optimisation_priority: list[PrioritisedSkill] = Field(
        default_factory=list,
        description="Ordered list of what to fix first",
    )
    seniority_calibration: str = Field(
        default="",
        description="Assessment of resume seniority vs job seniority",
    )
    recommended_skill_order: list[str] = Field(
        default_factory=list,
        description="Skills reordered by priority for this specific job",
    )
    section_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-section match scores: summary, experience, skills, education",
    )


# ─── Agent 4: Bullet Optimiser ──────────────────────────────────────────────────


class OptimisedBullet(BaseModel):
    """A single rewritten bullet with audit trail."""

    role_index: int = Field(description="Index into ResumeData.roles")
    bullet_index: int = Field(description="Index into Role.bullets")
    original_text: str
    optimised_text: str
    keywords_injected: list[str] = []
    change_rationale: str = ""
    fabrication_flag: bool = Field(
        default=False,
        description="True if the change might add unverifiable claims",
    )
    quality_score: int = Field(default=5, description="Quality score 1-10 for the optimised bullet")
    improvements: list[str] = Field(default_factory=list, description="List of improvement tags applied: 'stronger_verb', 'added_metrics', 'keyword_injected', 'shortened', 'reframed'")


class OptimisedBullets(BaseModel):
    """Structured output from the Bullet Optimisation Agent."""

    mode_used: RewriteMode
    bullets: list[OptimisedBullet]
    summary_rewrite: str = Field(
        default="", description="Rewritten professional summary"
    )
    original_summary: str = ""


# ─── Agent 5: ATS Optimiser ─────────────────────────────────────────────────────


class ATSCheck(BaseModel):
    """Result of a single ATS compatibility check."""

    check_name: str
    passed: bool
    detail: str = ""


class ATSResume(BaseModel):
    """Structured output from the ATS Optimisation Agent."""

    markdown_content: str
    ats_checks: list[ATSCheck]
    keyword_density: dict[str, int] = Field(
        default_factory=dict,
        description="Keyword -> occurrence count in final text",
    )
    job_title_aligned: bool = False
    suggested_title: str = ""


# ─── Agent 6: Recruiter Optimiser ────────────────────────────────────────────────


class RecruiterResume(BaseModel):
    """Structured output from the Recruiter Optimisation Agent."""

    markdown_content: str
    narrative_improvements: list[str] = Field(
        default_factory=list,
        description="Summary of changes made for recruiter appeal",
    )
    leadership_signals_added: list[str] = []
    impact_enhancements: list[str] = []


# ─── Agent 7: Final Assembly ─────────────────────────────────────────────────────


class TalkingPoint(BaseModel):
    """An interview talking point derived from the tailored resume."""

    topic: str
    bullet_points: list[str]
    source_experience: str = Field(
        default="",
        description="Which role/bullet this maps back to",
    )


class MatchReport(BaseModel):
    """Comprehensive match report included in final output."""

    job_title: str
    company: str = ""
    overall_match_score: float
    cosine_similarity: float
    missing_keywords: list[str]
    keyword_frequency: list[KeywordFrequency] = []
    seniority_calibration: str = ""
    ats_checks: list[ATSCheck] = []
    rewrite_mode: str
    optimisation_summary: str = ""


class FinalOutput(BaseModel):
    """Complete output from the Final Assembly Agent."""

    ats_resume_md: str
    recruiter_resume_md: str
    match_report: MatchReport
    talking_points: list[TalkingPoint]
