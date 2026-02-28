# CVtailro

A production-grade, local, multi-agent resume tailoring system powered by Claude Code CLI.

CVtailro analyses job descriptions against your master resume (PDF or markdown), optimises for ATS compatibility and recruiter appeal, and outputs professionally formatted PDF resumes with match reports and interview talking points.

**No API key required** — uses your existing Claude Code CLI access.

## Architecture

7 specialised agents in a sequential pipeline:

```
resume.pdf ──> [1. Job Intelligence] ──> JobAnalysis
  job.txt  ──> [2. Resume Parser]    ──> ResumeData
                         ↓
               [3. Gap Analysis] ──> GapReport
                         ↓
               [4. Bullet Optimiser] ──> OptimisedBullets
                     ↓           ↓
           [5. ATS Optimiser]  [6. Recruiter Optimiser]
                     ↓           ↓
               [7. Final Assembly]
                         ↓
         tailored_resume_ats.pdf
         tailored_resume_recruiter.pdf
         match_report.json
         interview_talking_points.md
```

## Installation

### Prerequisites

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- WeasyPrint system dependencies (for PDF generation)

### macOS setup

```bash
# Install WeasyPrint system dependencies
brew install cairo pango gdk-pixbuf libffi

# Clone/enter the project
cd CVtailro

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Verify setup

```bash
# Check Claude CLI is available
claude --version

# Check Python deps
python -c "import pdfplumber, weasyprint, markdown, pydantic; print('All dependencies OK')"
```

## Usage

### Basic usage (conservative mode)

```bash
python orchestrator.py --job job.txt --resume resume.pdf
```

### Aggressive rewriting mode

```bash
python orchestrator.py --job job.txt --resume resume.pdf --mode aggressive
```

### With markdown resume

```bash
python orchestrator.py --job job.txt --resume master.md
```

### Verbose logging

```bash
python orchestrator.py --job job.txt --resume resume.pdf -v
```

### Re-run from a specific stage

```bash
python orchestrator.py --job job.txt --resume resume.pdf \
    --stage bullet_optimiser \
    --output-dir output/20260227_143000/
```

## Input Files

**resume** — Your master resume as a `.pdf`, `.md`, or `.txt` file. PDF text is automatically extracted. This is the source of truth that the system optimises from.

**job.txt** — The target job description as plain text or markdown. Copy-paste the full job posting.

## Output Files

All outputs are saved to a timestamped directory under `output/`:

| File | Description |
|------|-------------|
| `tailored_resume_ats.pdf` | Professionally formatted ATS-optimised resume PDF |
| `tailored_resume_recruiter.pdf` | Professionally formatted recruiter-optimised resume PDF |
| `tailored_resume_ats.md` | ATS version in markdown (for further editing) |
| `tailored_resume_recruiter.md` | Recruiter version in markdown (for further editing) |
| `match_report.json` | Match score, cosine similarity, missing keywords, ATS checks |
| `interview_talking_points.md` | STAR-format talking points for interview prep |
| `01_job_analysis.json` — `06_recruiter_resume.json` | Intermediate artifacts |
| `pipeline.log` | Full debug log |

## Conservative vs Aggressive Mode

**Conservative** (default):
- Subtle keyword swaps and reordering
- Preserves original sentence structure
- Auto-reverts any bullet flagged as potential fabrication

**Aggressive**:
- More significant rephrasing allowed
- Language elevation (e.g., "helped with" -> "spearheaded")
- Fabrication-flagged bullets are kept but flagged in the match report
- Still never fabricates metrics, tools, or experience

## How to Extend

### Add a new agent

1. Create a Pydantic model in `models.py` for the agent's output
2. Create a prompt template in `prompts/your_agent.txt`
3. Create `agents/your_agent.py` inheriting from `BaseAgent`
4. Set `PROMPT_FILE`, `OUTPUT_MODEL`, and `AGENT_NAME`
5. Implement `prepare_user_message()` and optionally `post_process()`
6. Add the stage to `orchestrator.py`

### Modify agent behaviour

Edit the prompt template in `prompts/`. No Python code changes needed.

### Customise PDF styling

Edit `RESUME_CSS` in `pdf_generator.py` to change fonts, colours, spacing, or layout.

## Project Structure

```
CVtailro/
├── orchestrator.py          # CLI entry point + pipeline
├── config.py                # Configuration (no API key needed)
├── models.py                # All Pydantic data contracts
├── base_agent.py            # Abstract base (uses Claude CLI)
├── similarity.py            # TF-IDF cosine similarity (pure Python)
├── pdf_generator.py         # HTML/CSS -> PDF via WeasyPrint
├── utils.py                 # File I/O, logging, PDF text extraction
├── agents/                  # 7 specialised agents
├── prompts/                 # Editable prompt templates
├── output/                  # Generated at runtime
└── requirements.txt         # pydantic + pdfplumber + weasyprint + markdown
```
