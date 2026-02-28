"""Abstract base class for all agents in the CVtailro pipeline.

Every agent inherits from BaseAgent, which provides:
- Prompt template loading from the prompts/ directory
- LLM calls via the OpenRouter API (OpenAI-compatible)
- JSON extraction and Pydantic validation
- Retry logic with exponential backoff
- Template variable substitution
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from analytics import pipeline_analytics
from config import AppConfig

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Module-level HTTP session for connection pooling ──────────────────────────
# Reusing a single Session across all agent calls avoids the overhead of
# establishing a new TCP + TLS connection on every request to OpenRouter.
_http_session = requests.Session()
_http_session.headers.update({
    "Content-Type": "application/json",
    "HTTP-Referer": "https://cvtailro.app",
    "X-Title": "CVtailro",
})


class AgentError(Exception):
    """Raised when an agent fails after all retries."""


def _mask_key(key: str) -> str:
    """Return a masked version of an API key for safe logging.

    Shows only the first 6 and last 3 characters, replacing the middle
    with '***'.  Short keys are fully masked.
    """
    if not key or len(key) <= 10:
        return "***"
    return f"{key[:6]}***{key[-3:]}"


class BaseAgent(ABC, Generic[T]):
    """Abstract base class that every CVtailro agent inherits from.

    Uses the OpenRouter API as the LLM backend.

    Subclasses must set:
        PROMPT_FILE: filename in the prompts/ directory
        OUTPUT_MODEL: the Pydantic model class for the output
        AGENT_NAME: human-readable name for logging
    """

    PROMPT_FILE: str = ""
    OUTPUT_MODEL: type[T] | None = None
    AGENT_NAME: str = ""
    AGENT_MAX_TOKENS: int = 8192  # Subclasses can override for smaller outputs

    MAX_RETRIES: int = 5
    RETRY_DELAY_BASE: float = 2.0
    API_TIMEOUT: int = 600  # 10 minutes per agent call

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._system_prompt: str | None = None

    @property
    def system_prompt(self) -> str:
        """Load and cache the system prompt from the prompts/ directory."""
        if self._system_prompt is None:
            prompt_path = Path(__file__).parent / "prompts" / self.PROMPT_FILE
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found: {prompt_path}"
                )
            self._system_prompt = prompt_path.read_text(encoding="utf-8")
        return self._system_prompt

    def format_system_prompt(self, **kwargs: Any) -> str:
        """Substitute template variables in the system prompt.

        Uses {variable_name} placeholders in prompt files.
        The {output_schema} placeholder is automatically populated
        with the JSON schema of the OUTPUT_MODEL.
        """
        prompt = self.system_prompt

        # Inject output schema if placeholder present
        if "{output_schema}" in prompt and self.OUTPUT_MODEL is not None:
            schema = json.dumps(
                self.OUTPUT_MODEL.model_json_schema(), indent=2
            )
            kwargs.setdefault("output_schema", schema)

        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    @abstractmethod
    def prepare_user_message(self, input_data: Any) -> str:
        """Format the input data into the user message string."""
        ...

    def post_process(self, parsed: T, input_data: Any) -> T:
        """Optional hook for post-LLM transformations."""
        return parsed

    def _call_llm_api(self, system_prompt: str, user_message: str) -> str:
        """Call the OpenRouter API and return the raw text response.

        Uses the module-level ``_http_session`` so that TCP/TLS connections
        to OpenRouter are reused across calls (connection pooling).

        Args:
            system_prompt: The system prompt string.
            user_message: The user message string.

        Returns:
            The LLM's text response.

        Raises:
            AgentError: On HTTP errors, timeouts, or empty responses.
        """
        # Only the Authorization header varies per call; the rest are
        # already set on the shared session.
        auth_header = {
            "Authorization": f"Bearer {self.config.api_key}",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.AGENT_MAX_TOKENS,
            "temperature": 0.3,
        }

        try:
            response = _http_session.post(
                OPENROUTER_URL,
                headers=auth_header,
                json=payload,
                timeout=self.API_TIMEOUT,
            )

            if response.status_code == 401:
                raise AgentError(
                    "Invalid OpenRouter API key. Check your key and try again."
                )
            if response.status_code == 402:
                raise AgentError(
                    "Insufficient OpenRouter credits. Add credits at openrouter.ai."
                )
            if response.status_code == 429:
                # Wait for rate limit to clear, then retry
                retry_after = int(response.headers.get("Retry-After", "10"))
                retry_after = min(retry_after, 30)  # Cap at 30s
                logger.warning(
                    f"Rate limited (429). Waiting {retry_after}s before retry..."
                )
                time.sleep(retry_after)
                raise AgentError(
                    "The AI service is handling high demand. Please wait a moment and try again."
                )
            if response.status_code >= 400:
                error_detail = ""
                try:
                    error_detail = response.json().get("error", {}).get("message", "")
                except (ValueError, json.JSONDecodeError):
                    error_detail = response.text[:200]
                raise AgentError(
                    f"OpenRouter API error {response.status_code}: {error_detail}"
                )

            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError) as e:
                raise AgentError(
                    f"OpenRouter returned non-JSON response: "
                    f"{response.text[:300]}"
                ) from e

            # Log token usage at DEBUG level
            usage = data.get("usage", {})
            if usage:
                logger.debug(
                    f"[{self.AGENT_NAME}] Token usage: "
                    f"prompt={usage.get('prompt_tokens', '?')}, "
                    f"completion={usage.get('completion_tokens', '?')}, "
                    f"total={usage.get('total_tokens', '?')}"
                )
                # Track analytics if this agent belongs to a pipeline job
                if self.config.job_id:
                    pipeline_analytics.record_api_call(
                        self.config.job_id, usage
                    )

            # Check if actual model differs from requested model
            actual_model = data.get("model", "")
            if actual_model and actual_model != self.config.model:
                logger.debug(
                    f"[{self.AGENT_NAME}] Model mismatch: "
                    f"requested={self.config.model}, actual={actual_model}"
                )

            choices = data.get("choices", [])
            if not choices:
                raise AgentError("OpenRouter returned no choices in response")

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise AgentError("OpenRouter returned empty content")

            return content

        except requests.exceptions.Timeout:
            raise AgentError(
                "The AI model took too long to respond. This usually resolves in a minute — please try again."
            )
        except requests.exceptions.ConnectionError:
            raise AgentError(
                "Could not reach the AI service. Please check your connection and try again."
            )

    def run(self, input_data: Any, **prompt_vars: Any) -> T:
        """Execute the full agent pipeline."""
        if self.OUTPUT_MODEL is None:
            raise AgentError(f"{self.AGENT_NAME}: OUTPUT_MODEL not set")

        system = self.format_system_prompt(**prompt_vars)
        user_message = self.prepare_user_message(input_data)

        logger.info(
            f"[{self.AGENT_NAME}] Calling OpenRouter API ({self.config.model})..."
        )
        logger.debug(
            f"[{self.AGENT_NAME}] API key (masked): {_mask_key(self.config.api_key)}"
        )
        logger.debug(
            f"[{self.AGENT_NAME}] System prompt: {len(system)} chars, "
            f"User message: {len(user_message)} chars"
        )

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                call_start = time.time()
                raw_text = self._call_llm_api(system, user_message)
                call_elapsed = time.time() - call_start
                logger.info(
                    f"[{self.AGENT_NAME}] LLM API call took {call_elapsed:.1f}s"
                )
                logger.debug(
                    f"[{self.AGENT_NAME}] Response length: {len(raw_text)} chars"
                )

                parsed_json = self._extract_json(raw_text)
                result = self.OUTPUT_MODEL.model_validate(parsed_json)
                result = self.post_process(result, input_data)

                logger.info(f"[{self.AGENT_NAME}] Completed successfully.")
                return result

            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                last_error = e
                logger.warning(
                    f"[{self.AGENT_NAME}] Attempt {attempt}/{self.MAX_RETRIES} "
                    f"failed (parse/validation): {e}"
                )
                if self.config.job_id:
                    pipeline_analytics.record_retry(self.config.job_id)
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE ** attempt
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

            except AgentError as e:
                last_error = e
                logger.warning(
                    f"[{self.AGENT_NAME}] Attempt {attempt}/{self.MAX_RETRIES} "
                    f"API error: {e}"
                )
                if self.config.job_id:
                    pipeline_analytics.record_retry(self.config.job_id)
                # Do NOT retry on auth errors — they won't self-resolve
                if "Invalid OpenRouter API key" in str(e) or "Insufficient" in str(e):
                    break
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE ** attempt
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    break

        raise AgentError(
            f"We couldn't complete this step after {self.MAX_RETRIES} attempts. "
            f"Please try again in a moment."
        ) from last_error

    @staticmethod
    def _fix_json(text: str) -> str:
        """Fix common LLM JSON quirks that cause parse failures.

        Handles: trailing commas, single-line // comments,
        unquoted NaN/Infinity, smart quotes.
        """
        # Remove single-line comments (// ...)
        text = re.sub(r"//[^\n]*", "", text)
        # Remove trailing commas before } or ]
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Replace smart quotes with straight quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        return text

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from LLM response text.

        Handles responses wrapped in ```json ... ``` code fences
        or bare JSON objects with nested braces. Automatically fixes
        common LLM JSON quirks (trailing commas, comments, etc.).
        """
        candidates: list[str] = []

        # Try to find JSON in code fence (greedy — handles nested objects)
        fence_match = re.search(
            r"```(?:json)?\s*(\{[^`]*\})\s*```", text, re.DOTALL
        )
        if fence_match:
            candidates.append(fence_match.group(1))

        # Try to find bare JSON object — outermost braces
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            candidates.append(text[brace_start : brace_end + 1])

        # Try each candidate: first raw, then with fixes applied
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                fixed = BaseAgent._fix_json(candidate)
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError(
            "No JSON object found in response", text, 0
        )
