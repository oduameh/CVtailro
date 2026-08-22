"""Unit tests for Stage-4 partial-failure salvage in BulletOptimiserAgent.

One failed role in the parallel fan-out must not kill the whole job — the
failed role's original bullets are kept as identity OptimisedBullet entries.
"""

from __future__ import annotations

import pytest

import base_agent
from agents.bullet_optimiser import BulletOptimiserAgent
from base_agent import AgentError
from config import AppConfig
from models import (
    BulletPoint,
    GapReport,
    OptimisedBullet,
    OptimisedBullets,
    ResumeData,
    RewriteMode,
    Role,
)


def _resume_data(num_roles: int = 3) -> ResumeData:
    roles = [
        Role(
            title=f"Engineer {i}",
            company=f"Co {i}",
            bullets=[
                BulletPoint(original_text=f"Bullet {i}-{j}") for j in range(2)
            ],
        )
        for i in range(num_roles)
    ]
    return ResumeData(name="Test User", roles=roles, global_skills=["python"], total_years_estimate=5.0)


def _gap_report() -> GapReport:
    return GapReport(match_score=50.0, cosine_similarity=0.4, missing_keywords=["docker"])


@pytest.fixture
def agent() -> BulletOptimiserAgent:
    return BulletOptimiserAgent(AppConfig(api_key="sk-test", model="openai/gpt-4o-mini"))


def _stub_role_call(monkeypatch, agent: BulletOptimiserAgent, failing_from: int) -> None:
    """Succeed for role 0 (returning optimised bullets), raise for the rest."""

    def fake_call(system_prompt, user_message, role_idx):
        if role_idx >= failing_from:
            raise AgentError(f"LLM outage for role {role_idx}")
        return OptimisedBullets(
            mode_used=RewriteMode.CONSERVATIVE,
            bullets=[
                OptimisedBullet(
                    role_index=0,
                    bullet_index=j,
                    original_text=f"Bullet 0-{j}",
                    optimised_text=f"Optimised 0-{j}",
                )
                for j in range(2)
            ],
            summary_rewrite="New summary",
        )

    monkeypatch.setattr(agent, "_call_for_single_role", fake_call)


@pytest.mark.unit
class TestParallelFanOutSalvage:
    """3+ roles use the parallel path; failures degrade per-role."""

    def test_failed_roles_keep_original_bullets(self, agent, monkeypatch):
        _stub_role_call(monkeypatch, agent, failing_from=1)

        result = agent.run({"resume_data": _resume_data(), "gap_report": _gap_report()})

        # 2 optimised from role 0 + 2 identity fallbacks each for roles 1 and 2
        assert len(result.bullets) == 6
        by_key = {(b.role_index, b.bullet_index): b for b in result.bullets}

        assert by_key[(0, 0)].optimised_text == "Optimised 0-0"
        assert result.summary_rewrite == "New summary"

        for idx in (1, 2):
            for bi in range(2):
                fallback = by_key[(idx, bi)]
                assert fallback.original_text == f"Bullet {idx}-{bi}"
                assert fallback.optimised_text == f"Bullet {idx}-{bi}"
                assert fallback.quality_score == 5
                assert "original bullet kept" in fallback.change_rationale

    def test_all_roles_failed_returns_all_identity_bullets(self, agent, monkeypatch):
        _stub_role_call(monkeypatch, agent, failing_from=0)

        result = agent.run({"resume_data": _resume_data(), "gap_report": _gap_report()})

        assert len(result.bullets) == 6
        assert all(b.optimised_text == b.original_text for b in result.bullets)

    def test_bullets_sorted_deterministically_after_salvage(self, agent, monkeypatch):
        _stub_role_call(monkeypatch, agent, failing_from=1)

        result = agent.run({"resume_data": _resume_data(), "gap_report": _gap_report()})

        keys = [(b.role_index, b.bullet_index) for b in result.bullets]
        assert keys == sorted(keys)

    def test_few_roles_still_use_single_call_path(self, agent, monkeypatch):
        sentinel = RewriteMode.CONSERVATIVE
        monkeypatch.setattr(base_agent.BaseAgent, "run", lambda self, data, **kw: sentinel)

        result = agent.run({"resume_data": _resume_data(num_roles=2), "gap_report": _gap_report()})

        assert result is sentinel
