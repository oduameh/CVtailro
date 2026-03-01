"""Cover Letter Generator Agent — Stage 7 of the pipeline.

Generates a tailored cover letter from the job analysis, tailored resume,
and candidate information. Produces both plain text and markdown formatted
versions of the cover letter.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from base_agent import BaseAgent
from models import JobAnalysis


class CoverLetterOutput(BaseModel):
    """Structured output from the Cover Letter Generator Agent."""

    cover_letter_text: str = Field(
        description="Plain text cover letter (no markdown formatting)"
    )
    cover_letter_md: str = Field(
        description="Markdown formatted cover letter"
    )


class CoverLetterAgent(BaseAgent["CoverLetterOutput"]):
    """Generate a tailored cover letter from job analysis and resume data."""

    PROMPT_FILE = "cover_letter.txt"
    OUTPUT_MODEL = CoverLetterOutput
    AGENT_NAME = "Cover Letter Generator"
    AGENT_MAX_TOKENS = 4096

    def prepare_user_message(self, input_data: Any) -> str:
        """Format job analysis, tailored resume, and candidate info for the LLM.

        Args:
            input_data: Dict with 'job_analysis' (JobAnalysis),
                       'resume_md' (str — the tailored resume markdown),
                       and 'candidate_name' (str) keys.
        """
        job: JobAnalysis = input_data["job_analysis"]
        resume_md: str = input_data["resume_md"]
        candidate_name: str = input_data.get("candidate_name", "")

        return (
            "TARGET JOB:\n"
            f"Title: {job.job_title}\n"
            f"Company: {job.company}\n"
            f"Required Skills: {json.dumps(job.required_skills)}\n"
            f"Preferred Skills: {json.dumps(job.preferred_skills)}\n"
            f"Responsibilities: {json.dumps(job.responsibilities)}\n\n"
            "CANDIDATE NAME:\n"
            f"{candidate_name or '[Your Name]'}\n\n"
            "TAILORED RESUME:\n"
            f"{resume_md}\n"
        )
