"""Pipeline analytics: token usage tracking and cost estimation.

Provides a module-level singleton ``pipeline_analytics`` that accumulates
token counts, API call counts, retry counts, and estimated USD cost across
all agent invocations within each pipeline job.  Thread-safe for use with
the concurrent stage execution in ``app.py``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class PipelineAnalytics:
    """Track token usage and timing per pipeline job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}  # job_id -> {tokens, timing, model, cost}
        self._global_stats: dict = {
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_jobs": 0,
            "jobs_by_model": defaultdict(int),
            "tokens_by_model": defaultdict(int),
        }

    # ── Job lifecycle ────────────────────────────────────────────────────────

    def start_job(self, job_id: str, model: str) -> None:
        """Register a new pipeline job before the first agent runs."""
        with self._lock:
            self._jobs[job_id] = {
                "model": model,
                "started_at": time.time(),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
                "retries": 0,
                "stages_completed": 0,
            }

    def record_api_call(self, job_id: str, usage_data: dict, model: str | None = None) -> None:
        """Record token usage from an OpenRouter API response.

        Args:
            job_id: The pipeline job identifier.
            usage_data: The ``usage`` dict from the OpenRouter response
                        (keys: ``prompt_tokens``, ``completion_tokens``).
            model: Optional model override (unused for now but kept for
                   future per-call model tracking).
        """
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            pt = usage_data.get("prompt_tokens", 0)
            ct = usage_data.get("completion_tokens", 0)
            job["prompt_tokens"] += pt
            job["completion_tokens"] += ct
            job["total_tokens"] += pt + ct
            job["api_calls"] += 1

    def record_retry(self, job_id: str) -> None:
        """Increment the retry counter for a pipeline job."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["retries"] += 1

    def complete_job(self, job_id: str) -> dict:
        """Finalise a job: compute duration, estimate cost, update globals.

        Returns:
            A snapshot dict of the completed job's stats, or ``{}`` if the
            job_id was not found.
        """
        with self._lock:
            if job_id not in self._jobs:
                return {}
            job = self._jobs[job_id]
            job["duration_seconds"] = time.time() - job["started_at"]

            # Estimate cost (rough OpenRouter pricing)
            model = job["model"]
            cost = self._estimate_cost(
                model, job["prompt_tokens"], job["completion_tokens"]
            )
            job["estimated_cost_usd"] = cost

            # Update global stats
            self._global_stats["total_tokens"] += job["total_tokens"]
            self._global_stats["total_cost_usd"] += cost
            self._global_stats["total_jobs"] += 1
            self._global_stats["jobs_by_model"][model] += 1
            self._global_stats["tokens_by_model"][model] += job["total_tokens"]

            return dict(job)

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_job_stats(self, job_id: str) -> dict:
        """Return a snapshot of a single job's current stats."""
        with self._lock:
            return dict(self._jobs.get(job_id, {}))

    def get_global_stats(self) -> dict:
        """Return aggregate stats across all completed jobs."""
        with self._lock:
            stats = dict(self._global_stats)
            stats["jobs_by_model"] = dict(stats["jobs_by_model"])
            stats["tokens_by_model"] = dict(stats["tokens_by_model"])
            # Compute average cost per job
            if stats["total_jobs"] > 0:
                stats["avg_cost_per_job"] = round(
                    stats["total_cost_usd"] / stats["total_jobs"], 6
                )
            else:
                stats["avg_cost_per_job"] = 0.0
            return stats

    # ── Cost estimation ──────────────────────────────────────────────────────

    @staticmethod
    def _estimate_cost(
        model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Rough cost estimate based on OpenRouter pricing.

        Prices are per **million** tokens (prompt / completion).
        """
        PRICING: dict[str, tuple[float, float]] = {
            # Free models
            "meta-llama/llama-3.3-70b-instruct:free": (0, 0),
            "qwen/qwen3-coder:free": (0, 0),
            "nousresearch/hermes-3-llama-3.1-405b:free": (0, 0),
            "openai/gpt-oss-120b:free": (0, 0),
            "mistralai/mistral-small-3.1-24b-instruct:free": (0, 0),
            "google/gemma-3-27b-it:free": (0, 0),
            "nvidia/nemotron-3-nano-30b-a3b:free": (0, 0),
            # Paid models
            "openai/gpt-4o-mini": (0.15, 0.60),
            "openai/gpt-4o": (2.50, 10.00),
            "anthropic/claude-sonnet-4": (3.00, 15.00),
            "google/gemini-2.5-pro-preview": (1.25, 10.00),
            "x-ai/grok-3": (3.00, 15.00),
            "deepseek/deepseek-chat-v3-0324": (0.27, 1.10),
            "meta-llama/llama-4-maverick": (0.50, 0.70),
        }
        prices = PRICING.get(model, (1.0, 3.0))  # default estimate
        cost = (
            prompt_tokens * prices[0] + completion_tokens * prices[1]
        ) / 1_000_000
        return round(cost, 6)


# Module-level singleton — imported by base_agent and app
pipeline_analytics = PipelineAnalytics()
