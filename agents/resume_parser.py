"""Agent 2: Resume Parsing Agent.

Parses a master resume in markdown format and extracts structured
information including roles, bullets, skills, metrics, and education.
"""

from __future__ import annotations

import re
from typing import Any

from base_agent import BaseAgent
from models import ResumeData


class ResumeParserAgent(BaseAgent["ResumeData"]):
    """Parses a markdown resume into structured data."""

    PROMPT_FILE = "resume_parser.txt"
    OUTPUT_MODEL = ResumeData
    AGENT_NAME = "Resume Parser Agent"
    AGENT_MAX_TOKENS = 8192

    def prepare_user_message(self, input_data: Any) -> str:
        """Format the raw resume markdown for the LLM.

        Args:
            input_data: Raw resume text in markdown format (str).
        """
        return f"Here is the master resume to parse:\n\n{input_data}"

    def post_process(self, parsed: ResumeData, input_data: Any) -> ResumeData:
        """Clean the raw_text_for_similarity field.

        Strips markdown formatting to produce clean text for
        cosine similarity computation.
        """
        raw_text = parsed.raw_text_for_similarity or str(input_data)

        # Remove markdown formatting
        cleaned = re.sub(r"[#*_`~\[\]()>|]", " ", raw_text)
        # Remove markdown links
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        parsed.raw_text_for_similarity = cleaned
        return parsed
