# CVtailro Architecture

This document describes the technical architecture for maintainers.

## Overview

CVtailro is a Flask-based SaaS that runs a 6-stage AI pipeline to tailor resumes for specific job descriptions. Users authenticate via Google OAuth or email/password, upload a PDF resume, paste a job description, and receive optimized outputs.

## Entry Points

| Entry | Purpose |
|-------|---------|
| `app.py` | Development server (`python app.py`) |
| `wsgi.py` | Gunicorn production entry (`gunicorn wsgi:application`) |
| `orchestrator.py` | CLI pipeline (no web server) |

## Configuration Layers

1. **Flask** -- `app/settings.py` -- env, DB, session, secrets
2. **Pipeline** -- `config.py` -- `AppConfig`, `RECOMMENDED_MODELS`, `DEFAULT_MODEL`
3. **Concurrency** -- `app/services/pipeline.py` -- `MAX_CONCURRENT_PIPELINES`, `MAX_QUEUE_DEPTH`, `JOB_TTL_SECONDS` (single source of truth)
4. **Admin** -- `app/services/admin_config.py` -- API key, model, rate limit (DB + `admin_config.json` fallback)
5. **Storage** -- `storage.py` -- R2 credentials from env
6. **Redis** -- `app/services/cache.py` -- `REDIS_URL` for rate limiting; graceful fallback when unset

## Module Layout

```
app/           Flask package (routes, services, models)
agents/        Pipeline agents (one per stage)
prompts/       LLM prompt templates (.txt)
config.py      Pipeline config
models.py      Pydantic schemas (JobAnalysis, ResumeData, etc.)
base_agent.py  OpenRouter client, retries, JSON extraction
storage.py     Cloudflare R2 client
pdf_generator  WeasyPrint, 3 CSS templates
docx_generator python-docx
```

## Route Surface

| Blueprint | Key routes |
|-----------|-----------|
| `main` | `/`, `/api/health`, `/api/status`, `/api/models` |
| `api` | `/api/tailor`, `/api/result`, `/api/progress`, `/api/download`, `/api/score-resume`, `/api/boost-bullet`, `/api/batch-tailor` |
| `auth` | Google OAuth + email/password login, register, verify, password reset |
| `admin` | `/admin` panel, config, users, analytics, diagnostics, observability hub (product usage, reliability, cost, audit, alerts, retention) |
| `history` | `/api/history` (paginated) |
| `saved_resumes` | `/api/saved-resumes` CRUD |
| `tracker` | `/api/tracker` -- job application tracking |
| `blog` | `/blog` -- content/SEO pages |

## Pipeline Stages

| Stage | Agent | Parallel? |
|-------|-------|------------|
| 1 | JobIntelligenceAgent | Yes (with 2) |
| 2 | ResumeParserAgent | Yes (with 1) |
| 3 | GapAnalysisAgent | No (pure Python) |
| 4 | BulletOptimiserAgent | Yes (per role if 3+) |
| 5 | ResumeOptimiserAgent | Yes (with 6) |
| 6 | TalkingPointsAgent (via FinalAssembly) | Yes (with 5) |

## Job Status Semantics

Canonical status values are defined in `app/models/job.py` as `JobStatus`:

| Status | Meaning |
|--------|---------|
| `running` | Pipeline in progress |
| `complete` | Pipeline finished successfully |
| `error` | Pipeline failed |

All queries (admin analytics, alerts, reliability, cost) must use `"complete"` -- never `"completed"`.

## Key Design Decisions

- **Unified resume** -- One optimized resume (not separate ATS/Recruiter). `recruiter_resume_md` column is deprecated (same as `ats_resume_md`).
- **3 PDF templates** -- Modern, Executive, Minimal -- user picks at download
- **Gap Analysis** -- Pure Python, no LLM (instant, cheap)
- **Bullet Optimiser** -- Parallel per role for speed
- **DB fallback** -- PDFs regenerated from stored markdown when R2/local missing
- **Admin config** -- DB-backed for multi-instance deployments
- **Telemetry** -- Analytics events with PII redaction, stored in `AnalyticsEvent` table

## Logging

- Web: `init_structured_logging()` in `app/middleware.py` (JSON in prod)
- CLI: `setup_logging()` in `utils.py` (console + per-job file)
- Logger names: `cvtailro.*` for app, `__name__` for agents

## Tests

179 tests across unit, integration, security, and observability suites. Run with `pytest tests/ -v`.

## Extending

1. **New prompt** -- Add `.txt` in `prompts/`, reference in agent
2. **New PDF template** -- Add CSS in `pdf_generator.py`, add to `TEMPLATES`
3. **New model** -- Add to `RECOMMENDED_MODELS` in `config.py`
4. **New route** -- Blueprint in `app/routes/`, register in `register_blueprints()`
5. **New DB column** -- Model in `app/models/`, `flask db migrate`
