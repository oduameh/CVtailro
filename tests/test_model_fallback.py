"""Free-model fallback chain in BaseAgent (config.FREE_FALLBACK_MODELS)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

import base_agent as base_agent_mod
from base_agent import AgentError, BaseAgent
from config import FREE_FALLBACK_MODELS, AppConfig


class _Out(BaseModel):
    value: str


class _DummyAgent(BaseAgent[_Out]):
    PROMPT_FILE = "final_assembly.txt"  # any existing prompt file
    OUTPUT_MODEL = _Out
    AGENT_NAME = "Dummy"

    def prepare_user_message(self, input_data):
        return "hello"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(base_agent_mod.time, "sleep", lambda _s: None)


def _agent(model: str) -> _DummyAgent:
    return _DummyAgent(AppConfig(api_key="sk-test", model=model))


def test_paid_model_has_no_fallback():
    agent = _agent("openai/gpt-4o-mini")
    assert agent._model_candidates() == ["openai/gpt-4o-mini"]


def test_free_model_candidates_walk_the_chain_without_duplicates():
    primary = FREE_FALLBACK_MODELS[1]
    candidates = _agent(primary)._model_candidates()
    assert candidates[0] == primary
    assert candidates.count(primary) == 1
    assert set(candidates) == set(FREE_FALLBACK_MODELS)


def test_free_model_falls_back_on_persistent_failure(monkeypatch):
    agent = _agent(FREE_FALLBACK_MODELS[0])
    calls: list[str] = []

    def fake_call(system, user, model=None):
        calls.append(model)
        if model == FREE_FALLBACK_MODELS[0]:
            raise AgentError("OpenRouter API error 500: boom")
        return '{"value": "ok"}'

    monkeypatch.setattr(agent, "_call_llm_api", fake_call)
    result = agent.run({})
    assert result.value == "ok"
    # Primary exhausted its full retry budget, then the first fallback won.
    assert calls[: agent.MAX_RETRIES] == [FREE_FALLBACK_MODELS[0]] * agent.MAX_RETRIES
    assert calls[agent.MAX_RETRIES] == FREE_FALLBACK_MODELS[1]


def test_model_unavailable_skips_straight_to_next_model(monkeypatch):
    agent = _agent(FREE_FALLBACK_MODELS[0])
    calls: list[str] = []

    def fake_call(system, user, model=None):
        calls.append(model)
        if model == FREE_FALLBACK_MODELS[0]:
            raise AgentError("Model unavailable: retired")
        return '{"value": "ok"}'

    monkeypatch.setattr(agent, "_call_llm_api", fake_call)
    assert agent.run({}).value == "ok"
    # No retry burn on a dead model — one attempt then switch.
    assert calls == [FREE_FALLBACK_MODELS[0], FREE_FALLBACK_MODELS[1]]


def test_daily_free_cap_aborts_whole_chain(monkeypatch):
    agent = _agent(FREE_FALLBACK_MODELS[0])
    calls: list[str] = []

    def fake_call(system, user, model=None):
        calls.append(model)
        raise AgentError("Daily free-model limit reached on the OpenRouter account.")

    monkeypatch.setattr(agent, "_call_llm_api", fake_call)
    with pytest.raises(AgentError, match="Daily free-model limit"):
        agent.run({})
    # Account-wide cap: switching free models cannot help, so only one call.
    assert calls == [FREE_FALLBACK_MODELS[0]]


def test_paid_model_failure_does_not_touch_free_chain(monkeypatch):
    agent = _agent("openai/gpt-4o-mini")
    calls: list[str] = []

    def fake_call(system, user, model=None):
        calls.append(model)
        raise AgentError("OpenRouter API error 500: boom")

    monkeypatch.setattr(agent, "_call_llm_api", fake_call)
    with pytest.raises(AgentError):
        agent.run({})
    assert set(calls) == {"openai/gpt-4o-mini"}
