"""Integration smoke tests for the pipeline orchestrator.

These exercise the real orchestration (stage threading, gap analysis, score
recompute, match-report assembly, lazy PDF generation, JobFile manifest, DB
persistence, status transitions) while mocking the LLM-backed agents at the
``.run()`` boundary so no network/model is touched.
"""

import queue

import pytest

from app.models import TailoringJob, User
from app.models.job import JobFile
from models import (
    ATSCheck,
    ATSResume,
    JobAnalysis,
    OptimisedBullet,
    OptimisedBullets,
    ResumeData,
    RewriteMode,
    Role,
    SeniorityLevel,
    TalkingPoint,
    TitleAlignment,
)

JOB_ANALYSIS = JobAnalysis(
    job_title="Senior Python Engineer",
    company="Acme",
    required_skills=["Python", "Flask", "PostgreSQL"],
    preferred_skills=["Redis"],
    tools=["Docker"],
    seniority_signals=["senior"],
    inferred_seniority=SeniorityLevel.SENIOR,
    soft_skills=["communication"],
    domain_keywords=["backend"],
    responsibilities=["Build APIs", "Scale services"],
    inferred_priority_skills=[],
    raw_text_for_similarity="senior python engineer flask postgresql docker backend apis",
)

RESUME_DATA = ResumeData(
    name="Jane Doe",
    roles=[
        Role(
            title="Software Engineer",
            company="OldCo",
            bullets=[],
        )
    ],
    global_skills=["Python", "Flask"],
    total_years_estimate=5.0,
    raw_text_for_similarity="python flask developer building backend apis and services",
)

OPTIMISED_BULLETS = OptimisedBullets(
    mode_used=RewriteMode.CONSERVATIVE,
    bullets=[
        OptimisedBullet(
            role_index=0,
            bullet_index=0,
            original_text="Did stuff",
            optimised_text="Built Python Flask APIs serving 1M requests/day",
        )
    ],
)

ATS_RESUME = ATSResume(
    markdown_content=(
        "# Jane Doe\nSan Francisco | jane@example.com | 555-1234\n\n"
        "## Summary\nSenior Python engineer with Flask and PostgreSQL.\n\n"
        "## Experience\n### Software Engineer, OldCo\n"
        "- Built Python Flask APIs with Docker and Redis on PostgreSQL\n"
    ),
    ats_checks=[ATSCheck(check_name="Contact present", passed=True)],
    title_alignments=[
        TitleAlignment(
            role_index=0, original_title="Software Engineer", aligned_title="Senior Python Engineer"
        )
    ],
)

TALKING_POINTS = [TalkingPoint(topic="Scaling APIs", bullet_points=["Scaled to 1M req/day"])]


class _CoverLetterResult:
    cover_letter_md = "Dear Hiring Manager,\n\nI am excited to apply.\n\nBest,\nJane"


@pytest.fixture
def patched_agents(monkeypatch):
    """Mock all LLM-backed agents; GapAnalysisAgent runs for real (pure Python)."""
    from agents.bullet_optimiser import BulletOptimiserAgent
    from agents.cover_letter import CoverLetterAgent
    from agents.final_assembly import FinalAssemblyAgent
    from agents.job_intelligence import JobIntelligenceAgent
    from agents.resume_optimiser import ResumeOptimiserAgent
    from agents.resume_parser import ResumeParserAgent

    monkeypatch.setattr(JobIntelligenceAgent, "run", lambda self, *a, **k: JOB_ANALYSIS)
    monkeypatch.setattr(ResumeParserAgent, "run", lambda self, *a, **k: RESUME_DATA)
    monkeypatch.setattr(BulletOptimiserAgent, "run", lambda self, *a, **k: OPTIMISED_BULLETS)
    monkeypatch.setattr(ResumeOptimiserAgent, "run", lambda self, *a, **k: ATS_RESUME)
    monkeypatch.setattr(FinalAssemblyAgent, "_generate_talking_points", lambda self, *a, **k: TALKING_POINTS)
    monkeypatch.setattr(CoverLetterAgent, "run", lambda self, *a, **k: _CoverLetterResult())


@pytest.fixture
def render_counters(monkeypatch):
    """Count and stub PDF/DOCX renders (writing real bytes so files exist on disk)."""
    counters = {"pdf": [], "docx": []}

    def fake_pdf(md, path, template="modern"):
        counters["pdf"].append((str(path), template))
        with open(path, "wb") as f:
            f.write(b"%PDF-fake" + b"0" * 600)

    def fake_docx(md, path, template="modern"):
        counters["docx"].append((str(path), template))
        with open(path, "wb") as f:
            f.write(b"PK-fake-docx" + b"0" * 600)

    monkeypatch.setattr("app.services.pipeline.generate_resume_pdf", fake_pdf)
    monkeypatch.setattr("app.services.pipeline.generate_resume_docx", fake_docx)
    return counters


def _seed_job(job_id, output_dir, user_id=None):
    from app.services.pipeline import jobs, jobs_lock

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "queue": queue.Queue(),
            "output_dir": str(output_dir),
            "created_at": 0,
            "user_id": user_id,
            "result": None,
            "error": None,
        }
    return jobs[job_id]["queue"]


def _drain(q):
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


@pytest.mark.integration
class TestPipelineHappyPath:
    def test_lazy_generation_and_manifest(self, flask_app, db, tmp_path, patched_agents, render_counters):
        from app.services.pipeline import jobs, run_pipeline_job

        user = User(google_id="pipe-g-1", email="pipe@example.com", name="Pipe", is_admin=False)
        db.session.add(user)
        db.session.commit()

        resume_file = tmp_path / "resume.md"
        resume_file.write_text("# Jane Doe\nPython Flask developer", encoding="utf-8")

        job_id = "pipejob1234567890"
        q = _seed_job(job_id, tmp_path, user_id=user.id)

        run_pipeline_job(
            flask_app,
            job_id,
            str(resume_file),
            "We need a senior Python Flask engineer with PostgreSQL and Docker. " * 3,
            "conservative",
            "modern",
            tmp_path,
            "test-api-key",
            "test-model",
            user.id,
        )

        # Status transitioned to complete in both DB and in-memory.
        db_job = db.session.get(TailoringJob, job_id)
        assert db_job is not None
        assert db_job.status == "complete"
        assert db_job.ats_resume_md
        assert db_job.recruiter_resume_md  # stripped variant persisted
        assert db_job.cover_letter_md

        assert jobs[job_id]["status"] == "complete"
        events = _drain(q)
        assert events[-1] == {"status": "complete"}

        # LAZY PROOF: only the selected template PDF + resume DOCX were rendered.
        assert len(render_counters["pdf"]) == 1
        assert "Modern.pdf" in render_counters["pdf"][0][0]
        assert len(render_counters["docx"]) == 1

        # FULL MANIFEST: every canonical name has a JobFile row.
        rows = JobFile.query.filter_by(job_id=job_id).all()
        names = {r.filename for r in rows}
        assert len(rows) == 23  # 8 template + 8 recruiter PDFs + 5 core + 2 cover
        assert any("Modern.pdf" in n and "Recruiter" not in n for n in names)
        assert any("Recruiter_Modern.pdf" in n for n in names)
        assert any(n.endswith("Resume.docx") for n in names)
        assert any("Cover_Letter.pdf" in n for n in names)

        # Only eagerly-written files carry a size; the rest are virtual rows.
        sized = [r for r in rows if r.size_bytes]
        assert len(sized) == 5  # selected PDF, resume DOCX, resume.md, report.json, talking points

        # The in-memory result still advertises the full file list to the frontend.
        result_files = jobs[job_id]["result"]["files"]
        assert len(result_files) == 23

    def test_files_list_in_result(self, flask_app, db, tmp_path, patched_agents, render_counters):
        from app.services.pipeline import run_pipeline_job

        resume_file = tmp_path / "resume.md"
        resume_file.write_text("# Jane Doe\nPython Flask developer", encoding="utf-8")
        job_id = "pipejob2234567890"
        _seed_job(job_id, tmp_path)

        run_pipeline_job(
            flask_app,
            job_id,
            str(resume_file),
            "Senior Python Flask engineer PostgreSQL Docker. " * 3,
            "conservative",
            "tech",
            tmp_path,
            "k",
            "m",
            None,
        )
        # The eagerly-rendered PDF must match the *selected* template.
        assert "Tech.pdf" in render_counters["pdf"][0][0]


@pytest.mark.integration
class TestPipelineFailurePath:
    def test_stage_failure_sets_error_and_releases_semaphore(
        self, flask_app, db, tmp_path, patched_agents, render_counters, monkeypatch
    ):
        from agents.resume_optimiser import ResumeOptimiserAgent
        from app.services import pipeline as pmod
        from app.services.pipeline import run_pipeline_job

        def boom(self, *a, **k):
            raise RuntimeError("optimiser exploded")

        monkeypatch.setattr(ResumeOptimiserAgent, "run", boom)

        before = pmod.pipeline_semaphore._value

        resume_file = tmp_path / "resume.md"
        resume_file.write_text("# Jane Doe\nPython", encoding="utf-8")
        job_id = "pipejobfail123456"
        q = _seed_job(job_id, tmp_path)

        run_pipeline_job(
            flask_app,
            job_id,
            str(resume_file),
            "Senior Python Flask engineer PostgreSQL. " * 3,
            "conservative",
            "modern",
            tmp_path,
            "k",
            "m",
            None,
        )

        db_job = db.session.get(TailoringJob, job_id)
        assert db_job.status == "error"
        assert db_job.error_message

        events = _drain(q)
        assert events[-1]["status"] == "error"

        # Semaphore must be returned even on failure (finally block).
        assert pmod.pipeline_semaphore._value == before
