"""Agent 4: Bullet Optimisation Agent.

Rewrites resume bullets to improve keyword alignment, quantified impact,
and action verb strength. Supports conservative and aggressive modes.
Never fabricates experience.

Speed optimisation: when the resume contains 3+ roles, each role's bullets
are sent to the LLM in parallel via a ThreadPoolExecutor, then merged.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import ValidationError

from base_agent import AgentError, BaseAgent
from models import (
    GapReport,
    OptimisedBullet,
    OptimisedBullets,
    ResumeData,
    RewriteMode,
)

logger = logging.getLogger(__name__)

# Maximum parallel LLM calls for role-level splitting
_MAX_WORKERS = 4


class BulletOptimiserAgent(BaseAgent["OptimisedBullets"]):
    """Rewrites resume bullets for better job alignment."""

    PROMPT_FILE = "bullet_optimiser.txt"
    OUTPUT_MODEL = OptimisedBullets
    AGENT_NAME = "Bullet Optimiser Agent"
    AGENT_MAX_TOKENS = 8192

    # ── Original single-call user message (unchanged) ────────────────────────

    def prepare_user_message(self, input_data: Any) -> str:
        """Format resume data and gap report for the LLM.

        The agent receives the resume structure and gap analysis
        but NOT the raw job description. This prevents the agent
        from inventing job-specific experience.

        Args:
            input_data: Dict with 'resume_data' (ResumeData) and
                       'gap_report' (GapReport) keys.
        """
        resume_data: ResumeData = input_data["resume_data"]
        gap_report: GapReport = input_data["gap_report"]

        # Build a focused view of what needs optimisation
        # Compact format — just role/bullet index and text, no metadata bloat
        resume_bullets = []
        for role_idx, role in enumerate(resume_data.roles):
            bullets_text = "\n".join(
                f"  [{role_idx},{bi}] {b.original_text}"
                for bi, b in enumerate(role.bullets)
            )
            resume_bullets.append(f"ROLE {role_idx}: {role.title} @ {role.company}\n{bullets_text}")

        # Compact gap — just the keywords to target
        missing = ", ".join(gap_report.missing_keywords[:15])
        weak = ", ".join(gap_report.weak_alignment[:10])
        priority = ", ".join(
            f"{p.skill} ({p.priority.value})"
            for p in gap_report.optimisation_priority[:10]
        )

        return (
            "CURRENT RESUME BULLETS:\n"
            + "\n\n".join(resume_bullets)
            + f"\n\nCURRENT PROFESSIONAL SUMMARY:\n{resume_data.summary}"
            + f"\n\nKEYWORDS TO INJECT: {missing}"
            + f"\nWEAK AREAS TO STRENGTHEN: {weak}"
            + f"\nPRIORITY ORDER: {priority}"
        )

    # ── Per-role message builder (for parallel splitting) ────────────────────

    def _prepare_single_role_message(
        self,
        role_idx: int,
        role: Any,
        gap_report: GapReport,
        summary: str,
        include_summary: bool,
    ) -> str:
        """Build a user message scoped to a single role.

        Args:
            role_idx: The index of this role in the original resume.
            role: The Role object.
            gap_report: Full gap report (keywords / priorities).
            summary: The candidate's professional summary text.
            include_summary: If True, include the summary section so the LLM
                rewrites it.  Only the first role call should set this.
        """
        bullets_text = "\n".join(
            f"  [{role_idx},{bi}] {b.original_text}"
            for bi, b in enumerate(role.bullets)
        )

        missing = ", ".join(gap_report.missing_keywords[:15])
        weak = ", ".join(gap_report.weak_alignment[:10])
        priority = ", ".join(
            f"{p.skill} ({p.priority.value})"
            for p in gap_report.optimisation_priority[:10]
        )

        msg = (
            f"CURRENT RESUME BULLETS:\n"
            f"ROLE {role_idx}: {role.title} @ {role.company}\n{bullets_text}\n\n"
        )

        if include_summary:
            msg += f"CURRENT PROFESSIONAL SUMMARY:\n{summary}\n\n"
        else:
            msg += (
                "CURRENT PROFESSIONAL SUMMARY:\n"
                "(Summary handled separately — focus ONLY on the bullets above. "
                "You may return an empty string for summary_rewrite.)\n\n"
            )

        msg += (
            f"KEYWORDS TO INJECT: {missing}\n"
            f"WEAK AREAS TO STRENGTHEN: {weak}\n"
            f"PRIORITY ORDER: {priority}"
        )
        return msg

    # ── Single-role LLM call with retry (mirrors BaseAgent.run logic) ────────

    def _call_for_single_role(
        self,
        system_prompt: str,
        user_message: str,
        role_idx: int,
    ) -> OptimisedBullets:
        """Run the LLM call + parse loop for one role's bullets.

        This replicates the retry logic from ``BaseAgent.run`` but
        operates on a single user message rather than the full
        ``prepare_user_message`` output.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_text = self._call_llm_api(system_prompt, user_message)
                logger.debug(
                    f"[{self.AGENT_NAME}] Role {role_idx} response: "
                    f"{len(raw_text)} chars"
                )
                parsed_json = self._extract_json(raw_text)
                result = OptimisedBullets.model_validate(parsed_json)
                return result

            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                last_error = e
                logger.warning(
                    f"[{self.AGENT_NAME}] Role {role_idx} attempt "
                    f"{attempt}/{self.MAX_RETRIES} failed (parse): {e}"
                )
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE ** attempt
                    time.sleep(delay)

            except AgentError as e:
                last_error = e
                logger.warning(
                    f"[{self.AGENT_NAME}] Role {role_idx} attempt "
                    f"{attempt}/{self.MAX_RETRIES} API error: {e}"
                )
                if "Invalid OpenRouter API key" in str(e) or "Insufficient" in str(e):
                    break
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAY_BASE ** attempt
                    time.sleep(delay)
                else:
                    break

        raise AgentError(
            f"{self.AGENT_NAME} failed for role {role_idx} after "
            f"{self.MAX_RETRIES} attempts: {last_error}"
        )

    # ── Overridden run() with parallel splitting ─────────────────────────────

    def run(self, input_data: Any, **prompt_vars: Any) -> OptimisedBullets:
        """Execute the bullet optimiser, splitting roles in parallel when beneficial.

        Strategy:
        - 1-2 roles: use the original single-call approach (overhead of
          parallelism is not worthwhile).
        - 3+ roles: fan out one LLM call per role, run them concurrently,
          then merge the results into a single ``OptimisedBullets``.
          The professional summary is included only in the FIRST role's
          call.
        """
        if self.OUTPUT_MODEL is None:
            raise AgentError(f"{self.AGENT_NAME}: OUTPUT_MODEL not set")

        resume_data: ResumeData = input_data["resume_data"]
        gap_report: GapReport = input_data["gap_report"]
        num_roles = len(resume_data.roles)

        # ── Few roles: fall back to the original single-call path ────────
        if num_roles < 3:
            return super().run(input_data, **prompt_vars)

        # ── Many roles: parallel path ────────────────────────────────────
        system_prompt = self.format_system_prompt(**prompt_vars)

        logger.info(
            f"[{self.AGENT_NAME}] Splitting {num_roles} roles into "
            f"parallel LLM calls (max_workers={_MAX_WORKERS})..."
        )

        # Build per-role user messages
        role_messages: list[tuple[int, str]] = []
        for idx, role in enumerate(resume_data.roles):
            msg = self._prepare_single_role_message(
                role_idx=idx,
                role=role,
                gap_report=gap_report,
                summary=resume_data.summary,
                include_summary=(idx == 0),  # summary only in first call
            )
            role_messages.append((idx, msg))

        # Fan out
        merged_bullets: list[OptimisedBullet] = []
        summary_rewrite: str = ""
        mode_used: RewriteMode | None = None

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(
                    self._call_for_single_role,
                    system_prompt,
                    msg,
                    idx,
                ): idx
                for idx, msg in role_messages
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()  # raises on failure

                merged_bullets.extend(result.bullets)

                # Take the summary_rewrite from the first role's response
                if idx == 0 and result.summary_rewrite:
                    summary_rewrite = result.summary_rewrite

                # All calls should agree on mode; take the first non-None
                if mode_used is None:
                    mode_used = result.mode_used

        # Sort bullets by (role_index, bullet_index) for deterministic order
        merged_bullets.sort(
            key=lambda b: (b.role_index, b.bullet_index)
        )

        merged = OptimisedBullets(
            mode_used=mode_used or RewriteMode.CONSERVATIVE,
            bullets=merged_bullets,
            summary_rewrite=summary_rewrite,
        )

        logger.info(
            f"[{self.AGENT_NAME}] Merged {len(merged_bullets)} bullets "
            f"from {num_roles} parallel calls."
        )

        # Apply the same post-processing safety checks
        merged = self.post_process(merged, input_data)

        logger.info(f"[{self.AGENT_NAME}] Completed successfully.")
        return merged

    # ── Post-processing (unchanged) ──────────────────────────────────────────

    def post_process(
        self, parsed: OptimisedBullets, input_data: Any
    ) -> OptimisedBullets:
        """Apply safety checks on the optimised bullets.

        In conservative mode, automatically revert any bullet where
        fabrication_flag is True back to the original text.
        """
        resume_data: ResumeData = input_data["resume_data"]

        # Store original summary
        parsed.original_summary = resume_data.summary

        if parsed.mode_used == RewriteMode.CONSERVATIVE:
            reverted_count = 0
            for bullet in parsed.bullets:
                if bullet.fabrication_flag:
                    logger.warning(
                        f"Reverting fabrication-flagged bullet "
                        f"(role {bullet.role_index}, bullet {bullet.bullet_index}): "
                        f"'{bullet.optimised_text}' -> '{bullet.original_text}'"
                    )
                    bullet.optimised_text = bullet.original_text
                    bullet.keywords_injected = []
                    bullet.change_rationale = "Reverted: flagged as potential fabrication in conservative mode"
                    reverted_count += 1

            if reverted_count > 0:
                logger.info(
                    f"Reverted {reverted_count} fabrication-flagged bullets "
                    f"in conservative mode."
                )

        return parsed
