# Testing Guide

## Quick Start

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, no system deps)
pytest tests/ -m unit -v

# Integration tests only (needs cairo for PDF)
pytest tests/ -m integration -v

# With coverage
pytest tests/ --cov=app --cov=similarity --cov=resume_quality --cov=keyword_density --cov=email_templates --cov-report=term-missing
```

## Test Structure

| Directory | Purpose |
|-----------|---------|
| `tests/unit/` | Unit tests — isolated, fast, no DB/network |
| `tests/integration/` | Integration tests — API, DB, auth flows |
| `tests/test_*.py` | Root tests (auth, health, security, API) |

## Markers

- **`@pytest.mark.unit`** — isolated unit tests (similarity, resume_quality, usage, etc.)
- **`@pytest.mark.integration`** — integration tests (API flows, auth, DB)

## Troubleshooting

### Run tests in order

```bash
pytest tests/ -v --tb=long -x
```
- `x` stops on first failure
- `--tb=long` shows full tracebacks

### Run a single test file

```bash
pytest tests/unit/test_usage.py -v
pytest tests/test_security.py -v
```

### Run a single test

```bash
pytest tests/unit/test_usage.py::TestUsageTracker::test_check_and_record_blocks_when_over_limit -v
```

### Verbose output

```bash
pytest tests/ -v --tb=long -s
```
- `-s` shows print statements

### Coverage for specific module

```bash
pytest tests/ --cov=app.services.file_service --cov-report=term-missing
```

## CI Pipeline

GitHub Actions runs:

1. **lint** — ruff check + format
2. **unit** — unit tests only (fast feedback)
3. **integration** — integration tests (after unit passes)
4. **test-all** — full suite with coverage

On failure, CI logs show full tracebacks for troubleshooting.
