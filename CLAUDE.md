# CVtailro — Project Context

## What This Is
Production AI resume tailoring SaaS. Live at https://cvtailro-production.up.railway.app
Users sign in with Google, upload a PDF resume + paste a job description, get back an AI-optimized resume in 8 PDF templates + DOCX + match report + interview talking points.

**Business model**: Admin (Emmanuel) provides the OpenRouter API key. Users get the service for free. Admin pays API costs.

## Tech Stack
- Python 3.13, Flask (app factory + blueprints), SQLAlchemy, Pydantic
- PostgreSQL on Railway, Cloudflare R2 (file storage), Redis (optional)
- Google OAuth (Authlib + Flask-Login)
- OpenRouter API for LLM calls (default: GPT-4o-mini)
- WeasyPrint (PDF), python-docx (DOCX)
- Gunicorn (production WSGI), Docker + Railway deployment
- CSRF (Flask-WTF), rate limiting (Flask-Limiter), Sentry (optional)

## Architecture — 6-Stage Pipeline
```
Stages 1+2 PARALLEL: Job Intelligence + Resume Parser
Stage 3: Gap Analysis (pure Python, instant)
Stage 4: Bullet Optimiser (parallel per role if 3+ roles)
Stages 5+6 PARALLEL: Resume Optimiser + Talking Points
```
One unified resume output (was two ATS+Recruiter, merged). All 8 PDF templates generated per job.

## Documentation
- [README.md](README.md) — Public overview, quick start
- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical architecture for maintainers
- [SETUP.md](SETUP.md) — Infrastructure and deployment
- `/admin/docs` — Comprehensive HTML docs (accessible from admin panel sidebar)

## Project Structure
```
CVtailro/
  app/                         # Flask application package
    __init__.py                # create_app() factory, security headers, migrations
    extensions.py              # db, login_manager, migrate, oauth, csrf, limiter
    settings.py                # Dev/Prod/Testing config classes
    middleware.py              # Structured logging, request IDs, Sentry
    models/                    # SQLAlchemy ORM models
      user.py                  # User (Google OAuth)
      job.py                   # TailoringJob, JobFile, JobApplication, JobStatus
      saved_resume.py          # SavedResume
      admin_config.py          # AdminSetting (DB-backed config)
    routes/                    # Flask blueprints
      main.py                  # /, /api/health, /api/status, /api/models
      api.py                   # /api/tailor, /api/result, /api/progress, /api/download, /api/score-resume, /api/boost-bullet, /api/batch-tailor
      auth.py                  # Google OAuth + email/password login/register
      admin.py                 # /admin panel, config, analytics, observability hub
      history.py               # /api/history
      saved_resumes.py         # /api/saved-resumes CRUD
      tracker.py               # /api/tracker — job application tracking
      blog.py                  # /blog — content pages
    services/                  # Business logic
      pipeline.py              # 6-stage pipeline orchestration, job storage, semaphore
      file_service.py          # Download: local → R2 → DB fallback
      admin_config.py          # AdminConfigManager (DB-first, file fallback)
      usage.py                 # UsageTracker, LoginRateLimiter
      cache.py                 # Redis wrapper with graceful fallback
      telemetry.py             # Analytics event tracking, PII redaction
  agents/                      # Pipeline agents (unchanged)
  prompts/                     # LLM prompt templates (unchanged)
  config.py                    # Pipeline AppConfig, RECOMMENDED_MODELS
  models.py                    # Pydantic schemas (JobAnalysis, ResumeData, etc.)
  base_agent.py                # OpenRouter API calls, retries, JSON extraction
  similarity.py                # TF-IDF cosine similarity
  analytics.py                 # Token/cost tracking singleton
  storage.py                   # Cloudflare R2 client
  pdf_generator.py             # 8 CSS templates, markdown→HTML→PDF
  docx_generator.py            # Markdown→DOCX (Calibri)
  resume_quality.py            # Pure Python resume quality scoring
  email_templates.py           # Follow-up email templates
  keyword_density.py           # Keyword density analysis
  templates/                   # Jinja2 templates (index.html, admin.html)
  tests/                       # pytest suite (179 tests: unit, integration, security, observability)
  migrations/                  # Alembic (Flask-Migrate)
  wsgi.py                      # Gunicorn entry point
  .github/workflows/ci.yml     # GitHub Actions: lint + test
```

## Key Files (Pipeline)
| File | Purpose |
|------|---------|
| `app/services/pipeline.py` | Pipeline orchestration, job state, semaphore |
| `app/services/file_service.py` | Three-tier download (disk/R2/DB regeneration) |
| `agents/resume_optimiser.py` | Stage 5: unified resume |
| `agents/bullet_optimiser.py` | Stage 4: parallel per-role bullet rewriting |
| `prompts/resume_optimiser.txt` | Unified ATS+Recruiter prompt |
| `base_agent.py` | OpenRouter API calls, JSON fix, retries |
| `config.py` | AppConfig, RECOMMENDED_MODELS, DEFAULT_MODEL |

## Running Locally
```bash
cd /Users/emmanuel/Desktop/CVtailro
source .venv/bin/activate
python app.py          # http://localhost:5050 (dev server)
# or
python wsgi.py         # same, via WSGI entry point
# or
gunicorn wsgi:application --bind 0.0.0.0:5050  # production mode
```

## Running Tests
```bash
pytest tests/ -v       # 179 tests
ruff check .           # linting (matches CI)
```

## Deploying
Push to GitHub → Railway auto-deploys via Dockerfile (Gunicorn).
See SETUP.md for full infrastructure guide.

## Key Design Decisions
1. Unified resume (not separate ATS/Recruiter) — one great resume > two decent ones
2. All 8 templates generated per job — user picks when downloading
3. Gap Analysis is pure Python (no LLM) — instant, saves API costs
4. Bullet Optimiser splits by role (parallel) — 5min → 10sec
5. Pipeline semaphore: max 5 concurrent, queue 50
6. DB fallback: PDFs regenerated from stored markdown on demand
7. Smart filenames: `Software_Engineer_Google_Modern.pdf`
8. Post-tailoring score: shows before/after improvement
9. Conservative mode: no longer auto-reverts flagged bullets (reframing ≠ fabrication)
10. Free models unreliable — GPT-4o-mini is the default for reliability
11. App factory pattern — `create_app()` for testability and clean config
12. Admin config DB-backed — works across multiple instances/containers
13. Structured JSON logging in production — request IDs for tracing
14. Dark mode — CSS variables, localStorage persistence, system preference
15. CSRF exempt on admin_bp and api_bp — session-based auth, no form tokens needed
16. Env var `OPENROUTER_API_KEY` always overrides DB — ensures Railway key rotation works
17. CSRF cookie regeneration — auto-recovers when session loses its token

## How to Extend
1. Modify prompts: edit `.txt` files in `prompts/` — no code changes
2. Add PDF template: add CSS in `pdf_generator.py`, add to TEMPLATES dict
3. Add new model: add to RECOMMENDED_MODELS in `config.py`
4. New DB column: add to model in `app/models/`, run `flask db migrate`
5. New route: create blueprint in `app/routes/`, register in `app/routes/__init__.py`
6. New service: add to `app/services/`, import where needed
