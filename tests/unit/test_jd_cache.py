"""Unit tests for the in-process cache fallback and the JD-intelligence cache key."""

import pytest

from app.services import cache as cache_mod
from app.services.cache import cache_get, cache_set
from app.services.pipeline import _jd_cache_key


@pytest.fixture(autouse=True)
def _clear_mem_cache():
    """Isolate each test from the module-level in-process cache."""
    with cache_mod._mem_cache_lock:
        cache_mod._mem_cache.clear()
    yield
    with cache_mod._mem_cache_lock:
        cache_mod._mem_cache.clear()


@pytest.mark.unit
class TestInProcessCacheFallback:
    """cache_get/cache_set work without Redis via the bounded in-process store."""

    def test_set_then_get(self):
        assert cache_set("k1", "v1", ttl=60) is True
        assert cache_get("k1") == "v1"

    def test_missing_key_returns_none(self):
        assert cache_get("nope") is None

    def test_expired_entry_returns_none(self):
        cache_set("k2", "v2", ttl=0)
        # ttl=0 → expires_at == now; a later read is past expiry.
        assert cache_get("k2") is None

    def test_eviction_bounds_memory(self):
        for i in range(cache_mod._MEM_CACHE_MAX + 20):
            cache_set(f"key{i}", f"val{i}", ttl=3600)
        with cache_mod._mem_cache_lock:
            assert len(cache_mod._mem_cache) <= cache_mod._MEM_CACHE_MAX


@pytest.mark.unit
class TestJdCacheKey:
    """The cache key separates by both JD text and model."""

    def test_same_inputs_same_key(self):
        assert _jd_cache_key("some jd", "gpt-4o-mini") == _jd_cache_key("some jd", "gpt-4o-mini")

    def test_different_model_different_key(self):
        assert _jd_cache_key("jd", "model-a") != _jd_cache_key("jd", "model-b")

    def test_different_jd_different_key(self):
        assert _jd_cache_key("jd one", "m") != _jd_cache_key("jd two", "m")

    def test_key_is_namespaced(self):
        assert _jd_cache_key("jd", "m").startswith("jdintel:")


@pytest.mark.integration
class TestStage1CacheBehaviour:
    """Stage 1 reuses cached JobAnalysis on identical (JD, model); re-runs otherwise."""

    def _patch_pipeline(self, monkeypatch):
        """Mock all agents; return a dict counting JobIntelligenceAgent.run calls."""
        from agents.bullet_optimiser import BulletOptimiserAgent
        from agents.cover_letter import CoverLetterAgent
        from agents.final_assembly import FinalAssemblyAgent
        from agents.job_intelligence import JobIntelligenceAgent
        from agents.resume_optimiser import ResumeOptimiserAgent
        from agents.resume_parser import ResumeParserAgent
        from tests.integration.test_pipeline_orchestration import (
            ATS_RESUME,
            JOB_ANALYSIS,
            OPTIMISED_BULLETS,
            RESUME_DATA,
            TALKING_POINTS,
            _CoverLetterResult,
        )

        calls = {"stage1": 0}

        def counting_stage1(self, *a, **k):
            calls["stage1"] += 1
            return JOB_ANALYSIS

        monkeypatch.setattr(JobIntelligenceAgent, "run", counting_stage1)
        monkeypatch.setattr(ResumeParserAgent, "run", lambda self, *a, **k: RESUME_DATA)
        monkeypatch.setattr(BulletOptimiserAgent, "run", lambda self, *a, **k: OPTIMISED_BULLETS)
        monkeypatch.setattr(ResumeOptimiserAgent, "run", lambda self, *a, **k: ATS_RESUME)
        monkeypatch.setattr(
            FinalAssemblyAgent, "_generate_talking_points", lambda self, *a, **k: TALKING_POINTS
        )
        monkeypatch.setattr(CoverLetterAgent, "run", lambda self, *a, **k: _CoverLetterResult())
        monkeypatch.setattr(
            "app.services.pipeline.generate_resume_pdf", lambda md, p, template="modern": None
        )
        monkeypatch.setattr(
            "app.services.pipeline.generate_resume_docx", lambda md, p, template="modern": None
        )
        return calls

    def _run_once(self, flask_app, tmp_path, job_id, jd, model):
        import queue

        from app.services.pipeline import jobs, jobs_lock, run_pipeline_job

        out = tmp_path / job_id
        out.mkdir(exist_ok=True)
        resume = out / "resume.md"
        resume.write_text("# Jane\nPython Flask developer", encoding="utf-8")
        with jobs_lock:
            jobs[job_id] = {
                "status": "running",
                "queue": queue.Queue(),
                "output_dir": str(out),
                "created_at": 0,
                "user_id": None,
                "result": None,
                "error": None,
            }
        run_pipeline_job(flask_app, job_id, str(resume), jd, "conservative", "modern", out, "k", model, None)

    def test_identical_jd_skips_second_llm_call(self, flask_app, db, tmp_path, monkeypatch):
        calls = self._patch_pipeline(monkeypatch)
        jd = "Senior Python Flask engineer with PostgreSQL and Docker. " * 3

        self._run_once(flask_app, tmp_path, "jdcache0000000001", jd, "model-x")
        assert calls["stage1"] == 1

        self._run_once(flask_app, tmp_path, "jdcache0000000002", jd, "model-x")
        assert calls["stage1"] == 1  # served from cache — no new LLM call

    def test_different_model_re_runs(self, flask_app, db, tmp_path, monkeypatch):
        calls = self._patch_pipeline(monkeypatch)
        jd = "Senior Python Flask engineer with PostgreSQL and Docker. " * 3

        self._run_once(flask_app, tmp_path, "jdcache0000000003", jd, "model-a")
        self._run_once(flask_app, tmp_path, "jdcache0000000004", jd, "model-b")
        assert calls["stage1"] == 2  # different model key → re-run

    def test_corrupted_cache_falls_through(self, flask_app, db, tmp_path, monkeypatch):
        from app.services.cache import cache_set
        from app.services.pipeline import _jd_cache_key

        calls = self._patch_pipeline(monkeypatch)
        jd = "Senior Python Flask engineer with PostgreSQL and Docker. " * 3
        model = "model-corrupt"
        cache_set(_jd_cache_key(jd, model), "{not valid json", ttl=3600)

        self._run_once(flask_app, tmp_path, "jdcache0000000005", jd, model)
        assert calls["stage1"] == 1  # bad cache entry → LLM still ran
