# CVtailro Security Audit Report

**Date:** 2026-03-03
**Last reviewed:** 2026-03-03 (tech debt cleanup pass -- status values unified, dead code removed, test coverage expanded to 179 tests including security suite)
**Scope:** Application code, auth/sessions, API endpoints, file handling, config/deployment, dependencies
**Depth:** Deep (code review + abuse cases + test guidance)

---

## Attack Surface Map

```
                            Internet
                               |
                     +---------+---------+
                     |   Railway / Docker |
                     |  (ProxyFix trust)  |
                     +---------+---------+
                               |
              +----------------+----------------+
              |           Flask App              |
              |  (Gunicorn, 2 workers, 4 threads)|
              +--+-------+--------+--------+----+
                 |       |        |        |
           +-----+  +----+   +---+---+ +--+-----+
           |Auth |  |API  |  |Admin  | |Files   |
           |OAuth|  |Tail |  |Config | |Download|
           |Email|  |Score|  |Users  | |R2/DB   |
           +-----+  +----+   +------+ +--------+
              |         |         |         |
         +----+----+  +-+--+  +--+--+  +---+---+
         |Google   |  |LLM |  | DB  |  |  R2   |
         |Resend   |  |API |  |PgSQL|  |  S3   |
         +---------+  +----+  +-----+  +-------+
```

**Trust boundaries:**
- User input enters via: multipart upload (resume PDF), form text (job description), JSON bodies (tracker, saved resumes, profile, admin config)
- LLM output flows into: PDF/DOCX generation (markdown -> HTML -> WeasyPrint)
- Session cookies carry auth state for all blueprints
- Admin panel uses a separate session key (`admin_authenticated`)

---

## Findings by Severity

### CRITICAL

#### C1. Blanket CSRF Exemptions on All Session-Authenticated Blueprints

**Files:** `app/routes/auth.py:27`, `app/routes/admin.py:35`, `app/routes/api.py:39`, `app/routes/saved_resumes.py:12`, `app/routes/tracker.py:14`

**Issue:** Every blueprint that handles state-changing requests has `csrf.exempt()` applied at the blueprint level. Combined with session-cookie authentication (SameSite=Lax), any GET-triggered or cross-site POST from a same-site origin can execute actions on behalf of a logged-in user.

**Impact:** An attacker hosting a page on a same-site domain (or exploiting a subdomain) can:
- Change a user's password (OAuth-only users have no current password requirement -- see H1)
- Update profile name
- Create/delete saved resumes and tracker entries
- Trigger tailoring jobs (consuming API credits)
- If admin is logged in: change API keys, promote users to admin, change config

**Attack scenario:**
```html
<!-- Attacker page -->
<script>
fetch('https://cvtailro-production.up.railway.app/auth/profile/change-password', {
  method: 'POST',
  credentials: 'include',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({new_password: 'attacker123'})
});
</script>
```

**Remediation:**
1. Remove blanket `csrf.exempt()` from all blueprints.
2. For SPA JSON endpoints, use a custom CSRF scheme: send the CSRF token via a response cookie and require the frontend to echo it in a custom header (e.g. `X-CSRF-Token`). The Same-Origin Policy prevents cross-origin sites from reading/setting this header.
3. For the OAuth callback and SSE endpoints specifically, exempt only those individual routes.

---

#### C2. Admin Password Auto-Bootstrap: First Caller Wins

**File:** `app/routes/admin.py:68-74`

**Issue:** If `ADMIN_PASSWORD` is not set and no password exists in DB/file, the first POST to `/admin/api/login` sets the admin password to whatever the caller provides. On a fresh deployment (or after a DB wipe), any internet user can claim admin access.

**Attack scenario:** Bot scans for `/admin`, sends POST with any password, becomes admin.

**Remediation:**
1. Require `ADMIN_PASSWORD` as a mandatory env var in production; refuse to start if unset (or log a critical warning and disable the admin panel).
2. Alternatively, require the user to be a logged-in `is_admin` user before allowing password bootstrap.

---

### HIGH

#### H1. Password Change Without Current Password for OAuth Users (+ CSRF)

**File:** `app/routes/auth.py:336-357`

**Issue:** The `change_password` endpoint skips the current-password check when `current_user.password_hash` is `None` (OAuth-only accounts). Combined with C1 (no CSRF), this is a full account takeover: a CSRF attack can set a password on any OAuth user's account, then the attacker can log in via email/password.

**Remediation:**
1. Re-authenticate OAuth users before allowing password creation (e.g. require re-login via Google within the last 5 minutes).
2. At minimum, add CSRF protection to this endpoint (see C1).

---

#### H2. Hardcoded Default SECRET_KEY

**File:** `app/settings.py:28`

```python
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "cvtailro-dev-secret-change-in-production")
```

**Issue:** If `FLASK_SECRET_KEY` is not set in production, the default value is public (committed to source). Anyone can forge session cookies, CSRF tokens, and email verification/reset tokens (which use `itsdangerous` with the same `SECRET_KEY`).

**Impact:** Full authentication bypass -- forge any user's session, forge admin session, forge password-reset tokens.

**Remediation:**
1. Remove the default fallback in `ProductionSettings`. Raise an error at startup if the key is not set:
```python
class ProductionSettings(BaseSettings):
    SECRET_KEY = os.environ["FLASK_SECRET_KEY"]  # crash if missing
```
2. Ensure Railway has a strong random key (64+ chars) set.

---

#### H3. Download-Check Endpoint Leaks Cross-User Job Metadata

**File:** `app/routes/api.py:490-535`

**Issue:** `/api/download-check/<job_id>` returns `db_status`, `db_user_id`, and `db_user_match` for any job ID without checking ownership. Any authenticated user can enumerate job IDs and learn whether they exist, their status, and the owning user's ID.

**Remediation:** Filter by `user_id=current_user.id` or return 404 for non-owned jobs.

---

### MEDIUM

#### M1. Missing Remember-Me Cookie Hardening

**File:** `app/settings.py` (missing), `app/routes/auth.py:123,209,235`

**Issue:** `login_user(user, remember=True)` is called on every login path but `REMEMBER_COOKIE_SECURE`, `REMEMBER_COOKIE_HTTPONLY`, and `REMEMBER_COOKIE_SAMESITE` are never set. Flask-Login defaults may not match the session cookie settings, weakening protections.

**Remediation:** Add to `BaseSettings`:
```python
REMEMBER_COOKIE_SECURE = True  # override to False in DevelopmentSettings
REMEMBER_COOKIE_HTTPONLY = True
REMEMBER_COOKIE_SAMESITE = "Lax"
REMEMBER_COOKIE_DURATION = timedelta(days=14)
```

---

#### M2. HTML/Script Injection in PDF Rendering via `_inline_md`

**File:** `pdf_generator.py:758-763`

**Issue:** `_inline_md()` processes bold/italic/link markdown but does NOT escape the remaining text. LLM-generated resume content passes through this function and is rendered by WeasyPrint. If an LLM returns (or a user crafts) content containing HTML tags, those tags will be rendered in the PDF.

WeasyPrint fetches resources referenced in HTML (images, CSS). An `<img src="http://attacker.com/...">` tag in LLM output could trigger a server-side request (SSRF) during PDF generation.

**Remediation:**
1. Apply `_escape()` to the input text before processing markdown patterns in `_inline_md`:
```python
def _inline_md(text: str) -> str:
    text = _escape(text)  # escape HTML entities first
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+(?:\([^)]*\))*[^)]*)\)", r'<a href="\2">\1</a>', text)
    return text
```
2. Sanitize link `href` values to allow only `http://`, `https://`, and `mailto:` schemes.
3. Consider using a restricted URL fetcher for WeasyPrint that blocks non-HTTPS and private IP ranges.

---

#### M3. CSP Allows `unsafe-inline` for Scripts

**File:** `app/__init__.py:131-146`

**Issue:** `script-src 'self' 'unsafe-inline' ...` and `style-src 'self' 'unsafe-inline' ...` negate most of CSP's XSS protection. Any injection point that can write to an HTML response becomes exploitable.

**Remediation:**
1. Move inline scripts to external files.
2. Use nonce-based CSP (`script-src 'nonce-{random}'`) for any remaining inline scripts.
3. `style-src 'unsafe-inline'` is lower risk but should also be replaced with nonces or hashes where feasible.

---

#### M4. Admin API Key Stored in Plaintext (DB and File)

**File:** `app/services/admin_config.py:87-93`

**Issue:** The OpenRouter API key is stored as plaintext in both the `admin_settings` DB table and the `admin_config.json` file fallback. If the DB is compromised or the file is exposed, the key is directly readable.

**Remediation:**
1. Encrypt the API key at rest using a separate encryption key (e.g. `ENCRYPTION_KEY` env var) with `cryptography.fernet`.
2. At minimum, ensure `admin_config.json` is in `.gitignore` and `.dockerignore` (verify it is).
3. The DB approach is more secure than file (DB access is already gated), but at-rest encryption adds defense in depth.

---

#### M5. Download Endpoint Serves Any File in Job Output Directory

**File:** `app/services/file_service.py:44-49`

**Issue:** When a job is in memory, `send_from_directory` serves any file matching `Path(filename).name` under the output directory. This exposes `pipeline.log`, `input_job_description.txt`, and the original uploaded resume (`input_resume.pdf`) to anyone with the job ID.

**Remediation:** Add a filename allowlist:
```python
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".json"}
if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
    return jsonify({"error": "File type not allowed"}), 403
```
Or explicitly check against allowed filename patterns (e.g. must contain "resume", "talking", "cover", "match").

---

#### M6. Download Endpoint Has No Authentication or Rate Limiting

**File:** `app/routes/api.py:485-487`

**Issue:** `/api/download/<job_id>/<filename>` has no `@login_required` and no rate limiter. Anonymous jobs are accessible to anyone who knows the 16-char hex job ID. The DB fallback path regenerates PDF/DOCX on every request (CPU-heavy WeasyPrint rendering), making it a DoS vector.

**Remediation:**
1. Add `@limiter.limit("30 per minute")` to the download endpoint.
2. Cache regenerated PDFs to avoid repeated WeasyPrint invocations.
3. Consider requiring authentication for all downloads (with a short-lived download token for anonymous users).

---

#### M7. Batch Tailoring Skips PDF Validation

**File:** `app/routes/api.py:433-435`

**Issue:** `/api/batch-tailor` only checks file extension, not magic bytes or text extraction, unlike `/api/tailor` which validates both. A malformed or malicious file could reach the pipeline.

**Remediation:** Extract the PDF validation logic into a shared function and call it from both endpoints.

---

#### M8. ProxyFix Applied Unconditionally

**File:** `app/__init__.py:39-41`

**Issue:** `ProxyFix(x_for=1, x_proto=1, x_host=1, x_prefix=1)` trusts the first hop of `X-Forwarded-*` headers unconditionally. If the app is ever exposed directly (not behind a trusted reverse proxy), an attacker can spoof their IP address, affecting rate limiting and logging.

**Remediation:**
1. Only enable ProxyFix in production behind a known proxy.
2. Or document that direct exposure is not supported.

---

#### M9. Exception Details Leaked to Users

**Files:** `app/__init__.py:119`, `app/routes/api.py:112-113,396`

**Issue:** The global exception handler returns `str(e)` to API clients. Raw exception messages can reveal internal paths, library versions, or database details.

**Remediation:** Return a generic error message in production; log the full exception server-side:
```python
return _jsonify({"error": "Internal server error"}), 500
```

---

### LOW

#### L1. Reset Token in URL Query String

**File:** `app/routes/auth.py:283`

**Issue:** `redirect(f"/?reset_token={token}")` puts the token in the URL. It can leak via browser history, HTTP Referer header, or server access logs. Partially mitigated by `Referrer-Policy: strict-origin-when-cross-origin`.

**Remediation:** Use a short-lived session variable or POST-based flow to pass the token to the SPA.

---

#### L2. Dev Auth Bypass Route

**File:** `app/routes/auth.py:390-413`

**Issue:** `/auth/dev-login` bypasses OAuth when `DEV_AUTH_BYPASS=1`. If this env var is accidentally set in production, full auth bypass is available. The `DEV_AUTH_ADMIN=1` variant also grants admin.

**Remediation:**
1. Guard with `if app.debug and os.environ.get("DEV_AUTH_BYPASS") == "1"`.
2. Or remove entirely from production builds.

---

#### L3. Content-Disposition Header Injection

**File:** `app/services/file_service.py:117,169,186,210`

**Issue:** `filename` in `Content-Disposition` headers is not quoted or sanitized beyond `Path(filename).name`. Filenames containing special characters could cause parsing issues in some browsers.

**Remediation:** Use RFC 6266 compliant quoting:
```python
from urllib.parse import quote
headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
```

---

#### L4. Tracker Date Parsing Crashes on Invalid Input

**File:** `app/routes/tracker.py:65,87-91`

**Issue:** `datetime.fromisoformat(data["applied_date"])` raises `ValueError` on malformed dates, resulting in an unhandled 500 error.

**Remediation:** Wrap in try/except and return 400:
```python
try:
    applied_date = datetime.fromisoformat(data["applied_date"]) if data.get("applied_date") else None
except ValueError:
    return jsonify({"error": "Invalid date format"}), 400
```

---

#### L5. Dependencies Not Pinned; No Lock File

**File:** `requirements.txt`

**Issue:** Dependencies use range specifiers (e.g. `>=2.0.0,<3.0.0`). No `requirements.lock` or `pip-compile` output exists. Builds are not reproducible, and a compromised upstream package within the range could be pulled in.

**Remediation:**
1. Use `pip-compile` (pip-tools) to generate a `requirements.lock` with exact pins and hashes.
2. Pin `ruff` in CI to a specific version.
3. Add `pip-audit` to CI to catch known vulnerabilities.

---

#### L6. Docker Image Uses Floating Tag

**File:** `Dockerfile:1`

**Issue:** `FROM python:3.13-slim` uses a floating tag. A compromised or broken upstream image could affect builds.

**Remediation:** Pin to a digest:
```dockerfile
FROM python:3.13-slim@sha256:<digest>
```

---

#### L7. Docker Compose Weak Default Secrets

**File:** `docker-compose.yml:12-13`

**Issue:** `FLASK_SECRET_KEY` and `POSTGRES_PASSWORD` have weak defaults (`dev-secret-change-in-production` and `cvtailro`). If used in production without overrides, both are publicly known.

**Remediation:** Remove defaults for sensitive values; require explicit `.env` configuration.

---

#### L8. Admin Users Endpoint Dual Auth Check

**File:** `app/routes/admin.py:226-230`

**Issue:** `/admin/api/users` and `/admin/api/user-jobs/<id>` accept either session-based admin auth OR `is_admin` user auth, but this check is inline rather than using the `_admin_required` decorator. This dual-path is inconsistent and could introduce bypass if logic changes.

**Remediation:** Unify auth checks -- use a single decorator that accepts both auth methods.

---

## Summary Table

| ID  | Severity | Category | Issue |
|-----|----------|----------|-------|
| C1  | Critical | CSRF | Blanket CSRF exemptions on all blueprints |
| C2  | Critical | Auth | Admin password auto-bootstrap (first caller wins) |
| H1  | High | Auth | Password change without re-auth for OAuth users |
| H2  | High | Config | Hardcoded default SECRET_KEY |
| H3  | High | AuthZ | Download-check leaks cross-user job metadata |
| M1  | Medium | Session | Missing remember-me cookie hardening |
| M2  | Medium | Injection | HTML injection in PDF rendering via `_inline_md` |
| M3  | Medium | XSS | CSP allows `unsafe-inline` scripts |
| M4  | Medium | Secrets | Admin API key stored in plaintext |
| M5  | Medium | AuthZ | Download serves any file in job output dir |
| M6  | Medium | DoS | Download endpoint: no auth, no rate limit, regen on demand |
| M7  | Medium | Validation | Batch tailoring skips PDF magic-byte validation |
| M8  | Medium | Config | ProxyFix applied unconditionally |
| M9  | Medium | InfoLeak | Exception details leaked to users |
| L1  | Low | Token | Reset token in URL query string |
| L2  | Low | Auth | Dev auth bypass route in production |
| L3  | Low | Header | Content-Disposition header injection |
| L4  | Low | Validation | Tracker date parsing crashes on bad input |
| L5  | Low | Supply Chain | Dependencies not pinned; no lock file |
| L6  | Low | Supply Chain | Docker image floating tag |
| L7  | Low | Config | Docker Compose weak default secrets |
| L8  | Low | Auth | Inconsistent admin auth check pattern |

---

## Positive Observations

These security measures are already in place and working correctly:

- **Path traversal protection:** `Path(filename).name` strips directory components before file access
- **Upload size limit:** 10 MB `MAX_CONTENT_LENGTH`
- **Security headers:** X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy, HSTS (on HTTPS)
- **Session cookie settings:** HttpOnly, SameSite=Lax (session cookie)
- **Login brute-force protection:** `LoginRateLimiter` blocks after 5 failures for 15 minutes
- **Rate limiting on sensitive endpoints:** Registration (5/min), login (10/min), forgot-password (3/min), tailoring (10/hr), batch (3/hr)
- **ORM-only database access:** No raw SQL injection vectors found in user-facing code
- **Non-root Docker user:** App runs as uid 1000
- **Tini init:** Proper signal handling in container
- **Account enumeration prevention:** Registration/reset endpoints return generic messages
- **Sentry PII protection:** `send_default_pii=False`
- **OAuth state validation:** Handled by Authlib internally

---

## Remediation Priority

**Immediate (before next deploy):**
1. C2 -- Set `ADMIN_PASSWORD` env var on Railway; add startup check
2. H2 -- Verify `FLASK_SECRET_KEY` is set on Railway; remove default in production config

**This week:**
3. C1 -- Implement custom CSRF header scheme for SPA endpoints
4. H1 -- Require re-authentication before password creation for OAuth users
5. H3 -- Add ownership check to download-check endpoint
6. M6 -- Add rate limit to download endpoint

**Next sprint:**
7. M1 -- Add remember-me cookie settings
8. M2 -- Fix `_inline_md` to escape HTML before markdown processing
9. M5 -- Add filename allowlist for downloads
10. M7 -- Share PDF validation between single and batch tailor
11. M9 -- Suppress exception details in production responses

**Backlog:**
12. M3, M4, M8, L1-L8

---

## Test Plan

### CSRF Tests (validates C1 fix)
- [ ] POST to `/auth/profile/change-password` without CSRF header from cross-origin => expect 403
- [ ] POST to `/api/saved-resumes` without CSRF header => expect 403
- [ ] POST to `/admin/api/config` without CSRF header => expect 403
- [ ] POST to `/api/tracker` without CSRF header => expect 403
- [ ] Verify OAuth callback and SSE endpoints still work without CSRF token
- [ ] Verify SPA requests with correct CSRF header succeed

### Auth Tests (validates C2, H1, H2)
- [ ] Start app without `FLASK_SECRET_KEY` in production mode => expect startup error
- [ ] Start app without `ADMIN_PASSWORD` => verify admin panel is disabled or requires logged-in admin
- [ ] Attempt password change as OAuth user without re-auth => expect 403 or re-auth prompt
- [ ] Attempt admin login when no password is set => expect rejection (not auto-set)

### Access Control Tests (validates H3, M5, M6)
- [ ] GET `/api/download-check/<other-user-job-id>` => expect 404 (not metadata)
- [ ] GET `/api/download/<job-id>/pipeline.log` => expect 403 (not file contents)
- [ ] GET `/api/download/<job-id>/input_resume.pdf` => expect 403
- [ ] Rapid-fire 50 requests to `/api/download/...` => expect 429 after limit

### PDF Injection Tests (validates M2)
- [ ] Generate a resume with markdown containing `<img src="http://canary.example.com/x">` in a bullet => verify tag is escaped in HTML output (visible as text, not rendered)
- [ ] Generate a resume with `<script>alert(1)</script>` in a bullet => verify escaped

### Input Validation Tests (validates L4, M7)
- [ ] POST to `/api/tracker` with `applied_date: "not-a-date"` => expect 400 (not 500)
- [ ] POST to `/api/batch-tailor` with a non-PDF file renamed to `.pdf` => expect 400

### Rate Limit Tests (validates M6)
- [ ] Confirm `/api/download` returns 429 after exceeding rate limit
- [ ] Confirm PDF regeneration from DB is rate-limited
