# End-to-End Workflow Review — CVtailro

**Date:** 2025-03-02  
**Scope:** Full user journey from upload → pipeline → download, plus error paths and edge cases.

---

## Workflow Summary

```
User → /api/tailor (POST) → Pipeline (6 stages) → /api/progress (SSE) → /api/result → /api/download
         ↓                        ↓                      ↓                    ↓
    Validation              run_pipeline_job         EventSource           serve_download
    File save               (background thread)     or polling            (local/R2/DB)
```

---

## Issues Identified (Not Previously Logged)

### 1. **Output Directory Collision** — High severity

**Location:** `app/routes/api.py` L114, `utils.py` L175

**Problem:** `create_output_dir()` uses only a timestamp (`YYYYMMDD_HHMMSS`). Two requests in the same second get the same directory. The second job overwrites the first's `input_resume.pdf`, so the first job processes the wrong resume.

**Impact:** Wrong resume processed for one user; confusing/wrong results.

**Fix:** Include `job_id` in the path, e.g. `output/{timestamp}_{job_id}` or `output/{job_id}`.

---

### 2. **Match Report Download Fails (Case Sensitivity)** — Medium severity

**Location:** `app/services/file_service.py` L141

**Problem:** Pipeline saves `Match_Report.json` (e.g. `Software_Engineer_Google_Match_Report.json`). The DB fallback checks `"match_report" in safe_name`, which is case-sensitive. `"match_report"` is not in `"Match_Report.json"`, so the check fails and users get "File not found".

**Impact:** Match Report JSON cannot be downloaded via DB fallback (e.g. after R2/local cleanup).

**Fix:** Use `"match_report" in safe_name.lower()`.

---

### 3. **Download Ownership Inconsistency** — Medium severity

**Location:** `app/services/file_service.py` L37–59, L71–84

**Problem:** 
- **Tier 1 (local):** No ownership check — any `job_id` in `jobs` can be downloaded.
- **Tier 2 (R2):** Ownership checked only when `is_authed`; anonymous users can download any job's files.
- **Tier 3 (DB):** `_resolve_job_ownership` blocks anonymous users from authenticated users' jobs.

So anonymous users can access authenticated users' files via local or R2, but not via DB.

**Impact:** If `job_id` is known (e.g. URL sharing, guessing), anonymous users can download files for jobs that belong to authenticated users.

**Fix:** Apply the same ownership logic in all tiers (e.g. call `_resolve_job_ownership` before serving from local or R2).

---

### 4. **loadResults Retries on Error Status** — Low severity

**Location:** `templates/index.html` L3731–3738

**Problem:** When `d.status === "error"`, the code still retries up to 5 times with 2s delay instead of showing the error immediately.

**Impact:** User waits ~10 seconds before seeing the failure message.

**Fix:** Add `else if (d.status === "error") { showError(d.error); return; }` before the retry logic.

---

### 5. **Main Score Shows Original Instead of Tailored** — Low severity

**Location:** `templates/index.html` L3761

**Problem:** The main score ring uses `s.match_score`, which is the original score. The tailored score is in `s.tailored_match_score` and is only used in the before/after section.

**Impact:** The main result shows the pre-tailoring score instead of the improved score.

**Fix:** Use `(s.tailored_match_score ?? s.match_score)` for the main display.

---

### 6. **Poll Fallback Never Stops on 404** — Low severity

**Location:** `templates/index.html` L3697–3718

**Problem:** `pollForResult` only stops on `status === "complete"` or `status === "error"`. If the API returns 404 (e.g. job cleaned up), `d.status` is undefined and polling continues indefinitely.

**Impact:** Infinite polling when the job is no longer found.

**Fix:** Check `r.status === 404` and call `onError('Job not found.')` and stop polling.

---

### 7. **Temp File Leak on Regeneration Failure** — Low severity

**Location:** `app/services/file_service.py` L157–174, L176–191

**Problem:** In `_regenerate_pdf` and `_regenerate_docx`, if `generate_resume_pdf` or `generate_resume_docx` raises, `os.unlink(tmp_path)` is never called.

**Impact:** Temp files accumulate on repeated regeneration failures.

**Fix:** Use `try/finally` to always `os.unlink(tmp_path)` when the file exists.

---

### 8. **Local Output Never Cleaned When R2 Disabled** — Low severity

**Location:** `app/services/pipeline.py` L416–421, L546–550

**Problem:** `_cleanup_output` is only started when `r2_storage.is_configured`. In local/dev without R2, output directories are never deleted.

**Impact:** `output/` grows without bound in development or when R2 is not configured.

**Fix:** Run cleanup regardless of R2, or add a separate periodic cleanup for local output.

---

### 9. **setup_logging Adds Duplicate File Handlers** — Low severity

**Location:** `utils.py` L38–46, `app/services/pipeline.py` L139

**Problem:** Each pipeline job calls `setup_logging(log_file=output_dir / "pipeline.log")`, which adds a new `FileHandler` to the root logger. Handlers are never removed, so logs are duplicated across jobs.

**Impact:** Log bloat and possible performance impact over time.

**Fix:** Add and remove the file handler per job, or use a logger that is scoped to the job.

---

## Workflow Simulation — Happy Path

| Step | Component | Action | Status |
|------|-----------|--------|--------|
| 1 | User | Uploads PDF, pastes JD, clicks Tailor | OK |
| 2 | API | Validates file (magic bytes, pdfplumber), job text length | OK |
| 3 | API | Creates `output/{timestamp}/`, saves resume | ⚠️ Collision risk |
| 4 | API | Starts `run_pipeline_job` in daemon thread | OK |
| 5 | Pipeline | Stages 1+2 parallel → 3 → 4 → 5+6 parallel | OK |
| 6 | Frontend | SSE `/api/progress`, fallback to polling | OK |
| 7 | Pipeline | Persists to DB, uploads to R2 | OK |
| 8 | Frontend | Fetches `/api/result`, renders | ⚠️ Wrong main score |
| 9 | User | Clicks download | OK (local/R2) or ⚠️ Match Report via DB |

---

## Error Paths Reviewed

| Scenario | Handling | Notes |
|----------|----------|-------|
| Invalid PDF (image-based) | 400, user message | OK |
| Job description too short/long | 400, user message | OK |
| No API key | 400, admin message | OK |
| Queue full (503) | Frontend shows message | OK |
| Rate limit (429) | Frontend shows message | OK |
| Pipeline exception | DB status=error, in-memory error, pipeline_errors | OK |
| SSE disconnect | Retry 5×, then poll | OK |
| Job not found (404) | Poll never stops | ⚠️ Issue #6 |
| Job error status | loadResults retries 5× | ⚠️ Issue #4 |

---

## Recommendations

1. **Immediate:** Fix output dir collision (#1) and Match Report case sensitivity (#2). ✅ Fixed
2. **Short-term:** Align download ownership (#3), fix loadResults/poll behavior (#4, #6), and correct main score display (#5). ✅ Fixed
3. **Medium-term:** Temp file cleanup (#7), output cleanup when R2 disabled (#8), logging handler management (#9). ✅ Fixed

**All issues addressed as of 2025-03-02.**

---

## Files Touched in Review

- `app/routes/api.py` — tailor, progress, result, download
- `app/services/pipeline.py` — pipeline orchestration
- `app/services/file_service.py` — download tiers
- `app/services/usage.py` — rate limiting
- `utils.py` — create_output_dir, load_resume, setup_logging
- `templates/index.html` — frontend flow
