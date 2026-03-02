# CVtailro

AI-powered resume tailoring SaaS. Upload a PDF resume, paste a job description, and get an AI-optimized resume in 3 PDF templates (Modern/Executive/Minimal), DOCX, match report, and interview talking points.

**Live:** [cvtailro-production.up.railway.app](https://cvtailro-production.up.railway.app)

## Features

- **Google OAuth** — Sign in with your Google account
- **6-stage pipeline** — Job intelligence, resume parsing, gap analysis, bullet optimisation, resume optimisation, talking points
- **3 PDF templates** — Modern, Executive, Minimal — pick your style when downloading
- **Match report** — Before/after scores, missing keywords, section breakdown
- **Interview prep** — STAR-format talking points generated per job
- **OpenRouter** — Uses GPT-4o-mini by default (configurable via admin panel)

## Architecture

```
Stages 1+2 PARALLEL: Job Intelligence + Resume Parser
Stage 3: Gap Analysis (pure Python, instant)
Stage 4: Bullet Optimiser (parallel per role if 3+ roles)
Stages 5+6 PARALLEL: Resume Optimiser + Talking Points
```

One unified resume output. All 3 PDF templates generated per job.

## Quick Start

### Prerequisites

- Python 3.11+
- [WeasyPrint](https://doc.courtbouillon.org/weasyprint/) system dependencies (cairo, pango, gdk-pixbuf)

### macOS

```bash
brew install cairo pango gdk-pixbuf libffi
```

### Install and run

```bash
cd CVtailro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set env vars (see .env.example)
cp .env.example .env
# Edit .env with GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, OPENROUTER_API_KEY, etc.

python app.py   # or: python wsgi.py
# Open http://localhost:5050
```

### CLI (no web UI)

```bash
python orchestrator.py --job job.txt --resume resume.pdf
```

See [SETUP.md](SETUP.md) for full deployment (Railway, Google OAuth, R2, PostgreSQL).

## Project Structure

```
CVtailro/
├── app/                    # Flask application
│   ├── routes/             # API and web routes
│   ├── services/           # Pipeline, file service, admin config
│   └── models/             # SQLAlchemy ORM
├── agents/                 # Pipeline agents (6 stages)
├── prompts/                # LLM prompt templates
├── config.py               # Pipeline config, models
├── base_agent.py           # OpenRouter API client
├── pdf_generator.py        # 3 CSS templates → PDF
├── docx_generator.py       # Markdown → DOCX
├── app.py                  # Dev entry point
├── wsgi.py                 # Gunicorn entry point
└── orchestrator.py         # CLI pipeline
```

## Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -m unit -v             # Unit only (fast)
pytest tests/ -m integration -v     # Integration only
ruff check app/ tests/
```

See [TESTING.md](TESTING.md) for troubleshooting and CI pipeline details.

## Documentation

| File | Purpose |
|------|---------|
| [CLAUDE.md](CLAUDE.md) | Full project context for AI assistants |
| [SETUP.md](SETUP.md) | Infrastructure and deployment guide |
| [TESTING.md](TESTING.md) | Testing guide and troubleshooting |
