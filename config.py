"""Application configuration management for CVtailro."""

from __future__ import annotations

from dataclasses import dataclass

from models import RewriteMode

# All available models, grouped by free/paid.
# Display name -> OpenRouter model ID.
# Free models are labeled "(Free)" so users can identify them instantly.
RECOMMENDED_MODELS: dict[str, str] = {
    # ── Free models (no credits needed) ──────────────────────────
    "Qwen3 Coder 480B (Free)": "qwen/qwen3-coder:free",
    "Qwen3 Next 80B (Free)": "qwen/qwen3-next-80b-a3b-instruct:free",
    "Qwen3 235B Thinking (Free)": "qwen/qwen3-235b-a22b-thinking-2507",
    "Qwen3 VL 235B Thinking (Free)": "qwen/qwen3-vl-235b-a22b-thinking",
    "Qwen3 VL 30B Thinking (Free)": "qwen/qwen3-vl-30b-a3b-thinking",
    "Qwen3 4B (Free)": "qwen/qwen3-4b:free",
    "Hermes 3 Llama 405B (Free)": "nousresearch/hermes-3-llama-3.1-405b:free",
    "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Llama 3.2 3B (Free)": "meta-llama/llama-3.2-3b-instruct:free",
    "GPT-OSS 120B (Free)": "openai/gpt-oss-120b:free",
    "GPT-OSS 20B (Free)": "openai/gpt-oss-20b:free",
    "Step 3.5 Flash (Free)": "stepfun/step-3.5-flash:free",
    "NVIDIA Nemotron 3 Nano 30B (Free)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "NVIDIA Nemotron Nano 12B VL (Free)": "nvidia/nemotron-nano-12b-v2-vl:free",
    "NVIDIA Nemotron Nano 9B (Free)": "nvidia/nemotron-nano-9b-v2:free",
    "Mistral Small 3.1 24B (Free)": "mistralai/mistral-small-3.1-24b-instruct:free",
    "GLM 4.5 Air (Free)": "z-ai/glm-4.5-air:free",
    "Trinity Large Preview (Free)": "arcee-ai/trinity-large-preview:free",
    "Trinity Mini (Free)": "arcee-ai/trinity-mini:free",
    "Solar Pro 3 (Free)": "upstage/solar-pro-3:free",
    "Gemma 3 27B (Free)": "google/gemma-3-27b-it:free",
    "Gemma 3 12B (Free)": "google/gemma-3-12b-it:free",
    "Gemma 3 4B (Free)": "google/gemma-3-4b-it:free",
    "Gemma 3n 4B (Free)": "google/gemma-3n-e4b-it:free",
    "Gemma 3n 2B (Free)": "google/gemma-3n-e2b-it:free",
    "Dolphin Mistral 24B (Free)": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "LFM 2.5 1.2B Thinking (Free)": "liquid/lfm-2.5-1.2b-thinking:free",
    "LFM 2.5 1.2B (Free)": "liquid/lfm-2.5-1.2b-instruct:free",
    "Free Models Router (Auto)": "openrouter/free",
    # ── Paid models (requires credits) ───────────────────────────
    "Claude Sonnet 4": "anthropic/claude-sonnet-4",
    "Claude Opus 4": "anthropic/claude-opus-4",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "o3 Mini": "openai/o3-mini",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro-preview",
    "Grok 3": "x-ai/grok-3",
    "DeepSeek V3": "deepseek/deepseek-chat-v3-0324",
    "DeepSeek R1": "deepseek/deepseek-r1",
    "Llama 4 Maverick": "meta-llama/llama-4-maverick",
}

DEFAULT_MODEL = "qwen/qwen3-coder:free"


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
