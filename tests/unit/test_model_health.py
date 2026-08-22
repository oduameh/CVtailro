"""Unit tests for the BaseAgent model-health circuit breaker."""

from __future__ import annotations

import threading

import pytest
from pydantic import BaseModel

import base_agent as base_agent_mod
from base_agent import (
    MODEL_HEALTH_COOLDOWN_SECONDS,
    MODEL_HEALTH_FAILURE_THRESHOLD,
    AgentError,
    BaseAgent,
    _order_candidates_by_health,
    is_model_cooling,
    record_model_failure,
    record_model_success,
)
from config import FREE_FALLBACK_MODELS, AppConfig


@pytest.fixture(autouse=True)
def _clear_model_health():
    """Isolate each test from the module-level health registry."""
    with base_agent_mod._model_health_lock:
        base_agent_mod._model_health.clear()
    yield
    with base_agent_mod._model_health_lock:
        base_agent_mod._model_health.clear()


@pytest.mark.unit
class TestRecordModelFailure:
    """Failures below the threshold never cool; threshold arms the cooldown."""

    def test_below_threshold_does_not_cool(self):
        for i in range(MODEL_HEALTH_FAILURE_THRESHOLD - 1):
            record_model_failure("m/a", now=1000.0 + i)
        assert is_model_cooling("m/a", 1000.0 + MODEL_HEALTH_FAILURE_THRESHOLD) is False

    def test_reaching_threshold_cools_until_now_plus_cooldown(self):
        t0 = 1000.0
        record_model_failure("m/b", now=t0)
        record_model_failure("m/b", now=t0 + 5)
        assert is_model_cooling("m/b", t0 + 5) is True
        assert is_model_cooling("m/b", t0 + 5 + MODEL_HEALTH_COOLDOWN_SECONDS) is False

    def test_failures_keep_counting_past_threshold(self):
        record_model_failure("m/c", now=1000.0)
        record_model_failure("m/c", now=1010.0)
        record_model_failure("m/c", now=1020.0)
        entry = base_agent_mod._model_health["m/c"]
        assert entry["consecutive_failures"] == 3
        # Latest failure re-arms the window from the newest timestamp.
        assert entry["cooldown_until"] == pytest.approx(1020.0 + MODEL_HEALTH_COOLDOWN_SECONDS)

    def test_per_model_isolation(self):
        record_model_failure("m/x", now=1000.0)
        record_model_failure("m/y", now=1000.0)
        assert is_model_cooling("m/x", 1001.0) is False
        assert is_model_cooling("m/z", 1001.0) is False

    def test_unknown_model_is_never_cooling(self):
        assert is_model_cooling("never-seen", 12345.0) is False


@pytest.mark.unit
class TestRecordModelSuccess:
    """A healthy call fully resets the model's health entry."""

    def test_success_clears_cooldown(self):
        record_model_failure("m/a", now=1000.0)
        record_model_failure("m/a", now=1001.0)
        assert is_model_cooling("m/a", 1002.0) is True
        record_model_success("m/a")
        assert is_model_cooling("m/a", 1002.0) is False

    def test_success_restarts_failure_count(self):
        record_model_failure("m/b", now=1000.0)
        record_model_failure("m/b", now=1001.0)
        record_model_success("m/b")
        # Fresh start: one more failure is again below the threshold.
        record_model_failure("m/b", now=1002.0)
        assert is_model_cooling("m/b", 1003.0) is False
        assert base_agent_mod._model_health["m/b"]["consecutive_failures"] == 1

    def test_success_for_unknown_model_is_a_noop(self):
        record_model_success("ghost")
        assert base_agent_mod._model_health.get("ghost") is None


@pytest.mark.unit
class TestCooldownExpiry:
    """Cooldown expires exactly at cooldown_until (strictly-less comparison)."""

    def test_cooling_flips_off_after_expiry(self):
        t0 = 500.0
        for ts in (t0, t0 + 1):
            record_model_failure("m/a", now=ts)
        expiry = t0 + 1 + MODEL_HEALTH_COOLDOWN_SECONDS
        assert is_model_cooling("m/a", expiry - 1) is True
        assert is_model_cooling("m/a", expiry) is False

    def test_expired_model_can_trip_again(self):
        record_model_failure("m/b", now=0.0)
        record_model_failure("m/b", now=1.0)
        expiry = 1.0 + MODEL_HEALTH_COOLDOWN_SECONDS
        assert is_model_cooling("m/b", expiry) is False
        # After expiry the counter keeps counting, so one more failure stays hot.
        record_model_failure("m/b", now=expiry + 10)
        assert is_model_cooling("m/b", expiry + 10) is True


@pytest.mark.unit
class TestOrderCandidatesByHealth:
    """Candidate partitioning: healthy keep order first, cooling trail behind."""

    CANDIDATES = ["a/free", "b/free", "c/free", "d/free"]

    def _cool(self, *models: str) -> None:
        for m in models:
            record_model_failure(m, now=0.0)
            record_model_failure(m, now=1.0)

    def test_no_cooling_models_returns_original_order(self):
        healthy, cooling = _order_candidates_by_health(list(self.CANDIDATES), now=2.0)
        assert healthy == self.CANDIDATES
        assert cooling == []

    def test_all_cooling_returns_everything_as_fallback(self):
        self._cool(*self.CANDIDATES)
        healthy, cooling = _order_candidates_by_health(list(self.CANDIDATES), now=2.0)
        assert healthy == []
        assert cooling == self.CANDIDATES

    def test_mixed_partition_preserves_relative_order(self):
        self._cool("a/free", "c/free")
        healthy, cooling = _order_candidates_by_health(list(self.CANDIDATES), now=2.0)
        assert healthy == ["b/free", "d/free"]
        assert cooling == ["a/free", "c/free"]

    def test_primary_is_not_special_cased_when_cooling(self):
        self._cool(self.CANDIDATES[0])
        healthy, _cooling = _order_candidates_by_health(list(self.CANDIDATES), now=2.0)
        assert healthy[0] != self.CANDIDATES[0]

    def test_empty_candidate_list(self):
        assert _order_candidates_by_health([], now=2.0) == ([], [])


@pytest.mark.unit
class TestConcurrentRecording:
    """Hammering the registry from many threads must not lose updates."""

    THREADS = 8
    OPS_PER_THREAD = 50

    def test_parallel_failures_on_distinct_keys_are_exact(self):
        def worker(idx: int) -> None:
            key = f"m/worker-{idx}"
            for i in range(self.OPS_PER_THREAD):
                record_model_failure(key, now=float(i))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(self.THREADS):
            entry = base_agent_mod._model_health[f"m/worker-{i}"]
            assert entry["consecutive_failures"] == self.OPS_PER_THREAD
            assert entry["cooldown_until"] > 0.0

    def test_mixed_ops_on_shared_key_stay_consistent(self):
        stop = threading.Event()

        def hammer(key: str) -> None:
            while not stop.is_set():
                record_model_failure(key)
                record_model_success(key)

        keys = [f"m/shared-{i}" for i in range(4)]
        threads = []
        for key in keys:
            for _ in range(3):
                threads.append(threading.Thread(target=hammer, args=(key,), daemon=True))
        for t in threads:
            t.start()
        # Let them fight for a moment, then drain deterministically.
        stop.wait(0.05)
        for _t in threads:
            stop.set()
        for t in threads:
            t.join(timeout=5)
        assert not any(t.is_alive() for t in threads)

        for key in keys:
            with base_agent_mod._model_health_lock:
                entry = dict(base_agent_mod._model_health[key])
            assert isinstance(entry["consecutive_failures"], int)
            assert entry["consecutive_failures"] >= 0
            assert isinstance(entry["cooldown_until"], float)


# ── Wiring through BaseAgent.run() ────────────────────────────────────────────


class _Out(BaseModel):
    value: str


class _DummyAgent(BaseAgent[_Out]):
    PROMPT_FILE = "final_assembly.txt"  # any existing prompt file
    OUTPUT_MODEL = _Out
    AGENT_NAME = "Dummy"

    def prepare_user_message(self, input_data):
        return "hello"


@pytest.mark.unit
class TestRunDeprioritisesCooling:
    """run() tries healthy candidates before cooling ones; all-cooling still runs."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(base_agent_mod.time, "sleep", lambda _s: None)

    def _agent(self, model: str) -> _DummyAgent:
        return _DummyAgent(AppConfig(api_key="sk-test", model=model))

    def _trip_cooldown(self, model: str) -> None:
        # Real clock on purpose: cooldown windows are checked against
        # time.time(), so simulated epoch-1000 timestamps read as expired.
        record_model_failure(model)
        record_model_failure(model)

    def test_cooling_primary_is_tried_after_healthy_fallbacks(self, monkeypatch):
        primary = FREE_FALLBACK_MODELS[0]
        agent = self._agent(primary)
        self._trip_cooldown(primary)
        calls: list[str] = []

        def fake_call(system, user, model=None):
            calls.append(model)
            if model == FREE_FALLBACK_MODELS[2]:
                return '{"value": "ok"}'
            raise AgentError(f"OpenRouter API error 503: down ({model})")

        monkeypatch.setattr(agent, "_call_llm_api", fake_call)
        assert agent.run({}).value == "ok"
        # Cooling primary was pushed behind every healthy candidate — the job
        # succeeded on the first healthy fallback without ever touching it.
        assert calls[: agent.FALLBACK_RETRIES] == [FREE_FALLBACK_MODELS[1]] * agent.FALLBACK_RETRIES
        assert calls[-1] == FREE_FALLBACK_MODELS[2]
        assert primary not in calls

    def test_all_cooling_still_runs_the_chain(self, monkeypatch):
        agent = self._agent(FREE_FALLBACK_MODELS[0])
        for m in FREE_FALLBACK_MODELS:
            self._trip_cooldown(m)
        calls: list[str] = []

        def fake_call(system, user, model=None):
            calls.append(model)
            return '{"value": "ok"}'

        monkeypatch.setattr(agent, "_call_llm_api", fake_call)
        assert agent.run({}).value == "ok"
        assert calls[0] == FREE_FALLBACK_MODELS[0]  # original order preserved

    def test_chain_failure_error_names_tried_and_skipped_models(self, monkeypatch):
        primary = FREE_FALLBACK_MODELS[0]
        agent = self._agent(primary)
        self._trip_cooldown(FREE_FALLBACK_MODELS[1])

        def fake_call(system, user, model=None):
            raise AgentError(f"OpenRouter API error 503: down ({model})")

        monkeypatch.setattr(agent, "_call_llm_api", fake_call)
        with pytest.raises(AgentError) as exc_info:
            agent.run({})
        message = str(exc_info.value)
        assert "[tried:" in message
        assert primary in message
        assert "deprioritised while cooling" in message

    def test_account_cap_abort_records_no_model_failures(self, monkeypatch):
        agent = self._agent(FREE_FALLBACK_MODELS[0])
        calls: list[str] = []

        def fake_call(system, user, model=None):
            calls.append(model)
            raise AgentError("Daily free-model limit reached on the OpenRouter account.")

        monkeypatch.setattr(agent, "_call_llm_api", fake_call)
        with pytest.raises(AgentError, match="Daily free-model limit"):
            agent.run({})
        assert calls == [FREE_FALLBACK_MODELS[0]]
        # An account-wide cap says nothing about the model — nothing recorded.
        assert base_agent_mod._model_health == {}

    def test_successful_call_resets_health_via_run(self, monkeypatch):
        primary = FREE_FALLBACK_MODELS[0]
        agent = self._agent(primary)
        self._trip_cooldown(primary)

        def fake_call(system, user, model=None):
            if model == FREE_FALLBACK_MODELS[1]:
                return '{"value": "ok"}'
            raise AgentError(f"OpenRouter API error 503: down ({model})")

        monkeypatch.setattr(agent, "_call_llm_api", fake_call)
        assert agent.run({}).value == "ok"
        # The winning fallback's slate is wiped clean by its success.
        assert is_model_cooling(FREE_FALLBACK_MODELS[1]) is False
