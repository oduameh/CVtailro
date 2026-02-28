"""Application configuration management for CVtailro."""

from __future__ import annotations

from dataclasses import dataclass

from models import RewriteMode

# Curated models that reliably produce structured JSON output.
# Only 24B+ parameter models included — smaller ones fail on complex prompts.
RECOMMENDED_MODELS: dict[str, str] = {
    # ── Free models (no credits needed, 24B+ params) ─────────────
    "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Qwen3 Coder 480B (Free)": "qwen/qwen3-coder:free",
    "Hermes 3 Llama 405B (Free)": "nousresearch/hermes-3-llama-3.1-405b:free",
    "GPT-OSS 120B (Free)": "openai/gpt-oss-120b:free",
    "Mistral Small 3.1 24B (Free)": "mistralai/mistral-small-3.1-24b-instruct:free",
    "Gemma 3 27B (Free)": "google/gemma-3-27b-it:free",
    "NVIDIA Nemotron 30B (Free)": "nvidia/nemotron-3-nano-30b-a3b:free",
    # ── Paid models (requires credits) ───────────────────────────
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "GPT-4o": "openai/gpt-4o",
    "Claude Sonnet 4": "anthropic/claude-sonnet-4",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro-preview",
    "Grok 3": "x-ai/grok-3",
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick",
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
