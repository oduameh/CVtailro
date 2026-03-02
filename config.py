"""Application configuration management for CVtailro."""

from __future__ import annotations

from dataclasses import dataclass

from models import RewriteMode

# Curated models that reliably produce structured JSON output.
# Only 24B+ parameter models included — smaller ones fail on complex prompts.
RECOMMENDED_MODELS: dict[str, str] = {
    # ── Free models (no credits needed) ───────────────────────────
    # Best free options for structured JSON resume output, ranked by reliability
    "Qwen3 Coder 480B (Free)": "qwen/qwen3-coder:free",
    "Qwen3 Next 80B (Free)": "qwen/qwen3-next-80b-a3b-instruct:free",
    "Hermes 3 Llama 405B (Free)": "nousresearch/hermes-3-llama-3.1-405b:free",
    "GPT-OSS 120B (Free)": "openai/gpt-oss-120b:free",
    "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Trinity Large (Free)": "arcee-ai/trinity-large-preview:free",
    "NVIDIA Nemotron 30B (Free)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "StepFun 3.5 Flash (Free)": "stepfun/step-3.5-flash:free",
    "Mistral Small 3.1 24B (Free)": "mistralai/mistral-small-3.1-24b-instruct:free",
    "Gemma 3 27B (Free)": "google/gemma-3-27b-it:free",
    "Solar Pro 3 (Free)": "upstage/solar-pro-3:free",
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
    "Grok 3": "x-ai/grok-3",
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
            raise ValueError(
                "OpenRouter API key is required. "
                "Enter your key in the settings panel."
            )
        if not self.model or not self.model.strip():
            raise ValueError("Model ID is required.")
