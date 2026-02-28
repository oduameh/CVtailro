# CVtailro — Project Context

## What This Is

A production-grade, multi-agent resume tailoring system with a web UI. Users upload a PDF resume, paste a job description, and get back ATS-optimised and recruiter-optimised resumes as professionally formatted PDFs, plus a match report and interview talking points.

**Uses the OpenRouter API** as the LLM backend — supports 100+ models (Claude, GPT-4o, Gemini, Llama, etc.). Users provide their own API key via the web UI.

## Tech Stack

- Python 3.13, Flask, Pydantic
- WeasyPrint for PDF generation (requires system libs: cairo, pango, gdk-pixbuf)
- pdfplumber for PDF text extraction
- OpenRouter API for all LLM calls (via `requests` HTTP library)
- Docker support + Railway deployment

## Architecture

7 agents in a sequential pipeline with parallelization:

```
Stages 1+2 PARALLEL: Job Intelligence + Resume Parser
Stage 3: Gap Analysis (needs 1+2)
Stage 4: Bullet Optimiser (needs 2+3) — slowest stage (~3-5 min)
Stages 5+6 PARALLEL: ATS Optimiser + Recruiter Optimiser
Stage 7: Final Assembly + talking points
```

Each agent: loads a prompt template from `prompts/`, calls OpenRouter API, parses JSON response, validates with Pydantic, runs optional post_process().

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask web server, SSE progress streaming, routes |
| `orchestrator.py` | CLI entry point (alternative to web UI) |
| `base_agent.py` | Abstract base class — OpenRouter API calls, JSON extraction, retries |
| `models.py` | All Pydantic data contracts between agents |
| `config.py` | AppConfig dataclass, RECOMMENDED_MODELS, DEFAULT_MODEL |
| `pdf_generator.py` | Markdown→HTML→PDF with 3 templates (Executive/Modern/Minimal) |
| `similarity.py` | Pure Python TF-IDF cosine similarity |
| `utils.py` | File I/O, PDF extraction, logging |
| `agents/*.py` | 7 agent implementations |
| `prompts/*.txt` | Editable prompt templates with `{placeholder}` substitution |
| `templates/index.html` | Single-page web UI with settings panel |
| `railway.toml` | Railway deployment config |

## Running Locally

```bash
cd /Users/emmanuel/Desktop/CVtailro
source .venv/bin/activate
python app.py
# Open http://localhost:5050
# Enter your OpenRouter API key in the Settings panel
```

## Running via CLI

```bash
python orchestrator.py --job job.txt --resume resume.pdf --api-key sk-or-v1-...
python orchestrator.py --job job.txt --resume resume.pdf --model openai/gpt-4o
# Or set OPENROUTER_API_KEY env var instead of --api-key
```

## Running via Docker

```bash
docker build -t cvtailro .
docker run -p 5050:5050 cvtailro
```

## Deploying to Railway

Push to GitHub and connect the repo to Railway. The `railway.toml` handles build config. No server-side environment variables needed — API keys come from the user's browser.

## Known Issues & Current State

### Working
- Full pipeline end-to-end via web UI and CLI
- PDF generation with 3 templates (Executive, Modern, Minimal)
- Parallel stages 1+2 and 5+6
- PDF resume input + text/markdown JD input
- SSE real-time progress in browser
- Template selector in UI (Executive/Modern/Minimal toggle)
- Conservative vs Aggressive rewriting mode
- Multi-model support via OpenRouter (Claude, GPT-4o, Gemini, etc.)
- Settings panel with API key input + model selector (localStorage persistence)

### Known Issues
1. **Bullet Optimiser is slow** (~3-5 minutes). It's the most complex agent — rewrites every bullet with transferable skills framing. API timeout set to 600s.
2. **PDF parser is fragile** — handles multiple markdown formats but agents sometimes produce new patterns. The parser in `pdf_generator.py` has grown complex with many regex branches for: `# Name`, `## Section`, `### Role | Company | Location | Date`, `**Title** | Company | Location`, `**Degree** — School | Date`, categorized skills (`**Category:** items`), horizontal rules, etc.
3. **macOS port 5000** blocked by AirPlay — using port 5050 instead.
4. **Flask binds to 0.0.0.0** for Docker/Railway compatibility.

### Recent Improvements
- Migrated from Claude Code CLI to OpenRouter API (supports 100+ models)
- Added settings panel for API key and model selection
- Slimmed Docker image (removed Node.js and Claude CLI)
- Added Railway deployment support
- Prompts rewritten for transferable skills framing (not just keyword stuffing)
- Skills section capped at 20-30, grouped by category
- 2-page resume limit enforced in prompts
- QE fixes: thread safety, memory cleanup, file validation, SSE timeout, JSON regex
- UI: gradient header, progress bar, copy button, favicon, mobile responsive, template selector

## Prompt Engineering Notes

The quality of the output depends entirely on the prompt templates in `prompts/`. Key design decisions:

- **bullet_optimiser.txt**: Transferable skills strategy — reframes experience using target role's language. Includes good/bad rewrite examples. Anti-fabrication rules.
- **ats_optimiser.txt**: Hard 2-page limit. Skills capped at 20-30, grouped by `**Category:** items`. Job titles kept authentic.
- **recruiter_optimiser.txt**: Positions candidate for target role in summary without faking titles. "So what?" test on every bullet.
- All prompts inject `{output_schema}` — the Pydantic model's JSON schema.
- All prompts inject `{rewrite_mode}` for conservative/aggressive toggle.

## How to Extend

1. Add a new agent: create model in `models.py`, prompt in `prompts/`, agent in `agents/`, wire into `app.py` pipeline.
2. Modify agent behavior: edit the `.txt` file in `prompts/` — no code changes needed.
3. Add a PDF template: add CSS constant in `pdf_generator.py`, add to `TEMPLATES` dict, add option in `templates/index.html`.
4. Add a new model: add to `RECOMMENDED_MODELS` dict in `config.py` — the UI auto-populates from `/api/models`.
5. The `base_agent.py` handles all OpenRouter API interaction — agents just implement `prepare_user_message()` and optionally `post_process()`.
