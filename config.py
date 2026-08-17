"""Application configuration management for CVtailro."""

from __future__ import annotations

from dataclasses import dataclass

from models import RewriteMode

# Curated models that reliably produce structured JSON output.
# Free-model lineup rotates frequently on OpenRouter — /api/models cross-checks
# this dict against the live catalog and hides entries that have been retired.
RECOMMENDED_MODELS: dict[str, str] = {
    # ── Free models (no credits needed) ───────────────────────────
    # Best free options for structured JSON resume output, ranked by reliability
    "GLM 5.2 (Free)": "z-ai/glm-5.2:free",
    "Nemotron 3 Ultra 550B (Free)": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "Nemotron 3 Super 120B (Free)": "nvidia/nemotron-3-super-120b-a12b:free",
    "Nemotron 3.5 Lightning (Free)": "nvidia/nemotron-3.5-lightning:free",
    "Gemma 4 31B (Free)": "google/gemma-4-31b-it:free",
    "Gemma 4 26B (Free)": "google/gemma-4-26b-a4b-it:free",
    "Nemotron 3 Nano 30B (Free)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "Nemotron 3 Nano Omni (Free)": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "GPT-OSS 20B (Free)": "openai/gpt-oss-20b:free",
    "Laguna S 2.1 (Free)": "poolside/laguna-s-2.1:free",
    "North Mini Code (Free)": "cohere/north-mini-code:free",
    "Auto Router (Free)": "openrouter/free",
    # ── Paid models — Best Value ─────────────────────────────────
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "DeepSeek V3.2": "deepseek/deepseek-chat-v3-0324",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    # ── Paid models — High Quality ────────────────────────────────
    "GPT-4.1": "openai/gpt-4.1",
    "GPT-4o": "openai/gpt-4o",
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro-preview",
    # ── Paid models — Frontier ────────────────────────────────────
    "Claude Opus 4.6": "anthropic/claude-opus-4.6",
    "o4-mini": "openai/o4-mini",
    "Grok 4.6": "x-ai/grok-4.6",
}

DEFAULT_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration.

    Uses the OpenRouter API as the LLM backend.
    API key is provided per-request from the frontend.
    """

    rewrite_mode: RewriteMode = RewriteMode.CONSERVATIVE
    max_tokens: int = 16000
    output_dir: str | None = None
    verbose: bool = False
    api_key: str = ""
    model: str = DEFAULT_MODEL
    job_id: str = ""  # Set by app.py so agents can report analytics

    def validate_api_config(self) -> None:
        """Verify the API key and model are set.

        Raises:
            ValueError: If api_key is empty or model is empty.
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError("OpenRouter API key is required. Enter your key in the settings panel.")
        if not self.model or not self.model.strip():
            raise ValueError("Model ID is required.")
