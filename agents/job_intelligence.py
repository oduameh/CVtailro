"""Agent 1: Job Intelligence Agent.

Analyses a raw job description and extracts structured information
including required skills, preferred skills, tools, seniority signals,
and domain keywords.
"""

from __future__ import annotations

import re
from typing import Any

from base_agent import BaseAgent
from models import JobAnalysis


class JobIntelligenceAgent(BaseAgent["JobAnalysis"]):
    """Extracts structured job intelligence from a raw job description."""

    PROMPT_FILE = "job_intelligence.txt"
    OUTPUT_MODEL = JobAnalysis
    AGENT_NAME = "Job Intelligence Agent"
    AGENT_MAX_TOKENS = 4096

    def prepare_user_message(self, input_data: Any) -> str:
        """Format the raw job description text for the LLM.

        Args:
            input_data: Raw job description text (str).
        """
        return f"Here is the job description to analyse:\n\n{input_data}"

    def post_process(self, parsed: JobAnalysis, input_data: Any) -> JobAnalysis:
        """Clean the raw_text_for_similarity field using Python regex.

        Strips HTML tags, excessive whitespace, and common boilerplate
        patterns to produce clean text for cosine similarity computation.
        """
        raw_text = parsed.raw_text_for_similarity or str(input_data)

        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        # Remove URLs
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        parsed.raw_text_for_similarity = cleaned
        return parsed
