# CVtailro — Routes Map

All routes are Flask blueprints registered in `app/routes/__init__.py`. Templates are Jinja2, served from `templates/`.

---

## 1. Main Blueprint (`app/routes/main.py`)

| Method | URL            | Function       | Description                          | Auth     |
|--------|----------------|----------------|--------------------------------------|----------|
| GET    | `/`            | `index()`      | Renders `index.html` (main SPA)     | Public   |
| GET    | `/privacy`     | `privacy()`    | Renders `privacy.html`              | Public   |
| GET    | `/terms`       | `terms()`      | Renders `terms.html`                | Public   |
| GET    | `/contact`     | `contact()`    | Renders `contact.html`              | Public   |
| GET    | `/api/health`  | `health()`     | JSON health check (DB status)       | Public   |
| GET    | `/api/status`  | `api_status()` | JSON: configured? has_admin_password? | Public |
| GET    | `/api/models`  | `list_models()` | JSON: available models + default   | Public   |

---

## 2. API Blueprint (`app/routes/api.py`)

CSRF-exempt. Handles the core tailoring pipeline.

| Method | URL                                | Function            | Description                                  | Auth              |
|--------|------------------------------------|---------------------|----------------------------------------------|--------------------|
| POST   | `/api/tailor`                      | `start_tailoring()` | Upload resume + JD, start pipeline job       | Rate-limited (10/hr) |
| GET    | `/api/progress/<job_id>`           | `progress_stream()` | SSE stream of pipeline progress events       | Job owner or anon  |
| GET    | `/api/result/<job_id>`             | `get_result()`      | JSON result (scores, resume MD, files list)  | Job owner or anon  |
| GET    | `/api/download/<job_id>/<filename>`| `download_file()`   | Download PDF/DOCX file                       | Public (by job_id) |
| GET    | `/api/download-check/<job_id>`     | `download_check()`  | Debug endpoint: shows where files exist      | Login required     |

---

## 3. Auth Blueprint (`app/routes/auth.py`)

URL prefix: `/auth`. CSRF-exempt.

### Google OAuth

| Method | URL                    | Function           | Description                            |
|--------|------------------------|--------------------|----------------------------------------|
| GET    | `/auth/google/login`   | `google_login()`   | Initiates Google OAuth redirect        |
| GET    | `/auth/google/callback`| `google_callback()` | OAuth callback, creates/updates user  |

### Email/Password

| Method | URL                              | Function                | Description                          | Rate Limit     |
|--------|----------------------------------|-------------------------|--------------------------------------|----------------|
| POST   | `/auth/register`                 | `register()`            | Create email/password account        | 5/min          |
| POST   | `/auth/login`                    | `login()`               | Email/password login                 | 10/min         |
| GET    | `/auth/verify/<token>`           | `verify_email()`        | Email verification link handler      | —              |
| POST   | `/auth/resend-verification`      | `resend_verification()` | Resend verification email            | 3/min          |
| POST   | `/auth/forgot-password`          | `forgot_password()`     | Send password reset email            | 3/min          |
| GET    | `/auth/reset-password/<token>`   | `reset_password_page()` | Redirect to SPA with reset token     | —              |
| POST   | `/auth/reset-password`           | `reset_password()`      | Set new password with token          | 5/min          |

### Profile Management

| Method | URL                          | Function            | Description                     | Auth          |
|--------|------------------------------|---------------------|---------------------------------|---------------|
| POST   | `/auth/profile/update`       | `profile_update()`  | Update display name             | Login required |
| POST   | `/auth/profile/change-password` | `change_password()` | Change password (current + new) | Login required |

### Shared

| Method | URL               | Function     | Description                               | Auth |
|--------|-------------------|--------------|-------------------------------------------|------|
| POST   | `/auth/logout`    | `logout()`   | Clear session                             | Any  |
| GET    | `/auth/me`        | `me()`       | JSON: auth status, user info, provider    | Any  |
| GET    | `/auth/dev-login` | `dev_login()`| Dev-only: bypass OAuth (DEV_AUTH_BYPASS=1) | Dev  |

---

## 4. Admin Blueprint (`app/routes/admin.py`)

CSRF-exempt. Uses session-based admin auth (separate from user auth).

### Auth

| Method | URL                  | Function         | Description                    | Auth    |
|--------|----------------------|------------------|--------------------------------|---------|
| GET    | `/admin`             | `admin_page()`   | Renders `admin.html`           | Public  |
| POST   | `/admin/api/login`   | `admin_login()`  | Set/verify admin password      | 5/min   |
| POST   | `/admin/api/logout`  | `admin_logout()` | Clear admin session            | Any     |

### Configuration

| Method | URL                    | Function             | Description                       | Auth       |
|--------|------------------------|----------------------|-----------------------------------|------------|
| GET    | `/admin/api/config`    | `admin_get_config()` | Get masked API key, model, settings | Admin req. |
| POST   | `/admin/api/config`    | `admin_save_config()` | Save API key, model, rate limit  | Admin req. |
| POST   | `/admin/api/test-key`  | `admin_test_key()`   | Test OpenRouter API key validity  | Admin req. |

### Monitoring & Analytics

| Method | URL                          | Function              | Description                               | Auth       |
|--------|------------------------------|-----------------------|-------------------------------------------|------------|
| GET    | `/admin/api/stats`           | `admin_stats()`       | Jobs by status, time trends, success rate | Admin req. |
| GET    | `/admin/api/analytics`       | `admin_analytics()`   | Token usage, cost, jobs over time         | Admin req. |
| GET    | `/admin/api/live-stats`      | `admin_live_stats()`  | Active pipelines, memory, threads         | Admin req. |
| GET    | `/admin/api/recent-jobs`     | `admin_recent_jobs()` | Last 20 jobs (for dashboard)              | Admin req. |
| GET    | `/admin/api/errors`          | `admin_errors()`      | Pipeline error log                        | Admin req. |
| POST   | `/admin/api/errors/clear`    | `admin_clear_errors()`| Clear error log                           | Admin req. |
| GET    | `/admin/api/diagnostics`     | `admin_diagnostics()` | Health checks: DB, API, R2, Redis         | Admin req. |
| GET    | `/admin/api/usage`           | `admin_usage()`       | Rate limiting stats                       | Admin req. |

### User Management

| Method | URL                               | Function                | Description                      | Auth            |
|--------|------------------------------------|-------------------------|----------------------------------|-----------------|
| GET    | `/admin/api/users`                 | `admin_users()`         | All users with job counts        | Admin req.      |
| GET    | `/admin/api/user-jobs/<user_id>`   | `admin_user_jobs()`     | All jobs for a specific user     | Admin req.      |
| POST   | `/admin/api/users/<user_id>/admin` | `admin_set_user_admin()`| Promote/demote user admin status | Admin req.      |

---

## 5. History Blueprint (`app/routes/history.py`)

| Method | URL                      | Function            | Description                         | Auth           |
|--------|--------------------------|---------------------|-------------------------------------|----------------|
| GET    | `/api/history`           | `get_history()`     | Paginated job history for user      | Login required |
| GET    | `/api/history/<job_id>`  | `get_history_job()` | Detailed single job (all fields)    | Login required |

---

## 6. Saved Resumes Blueprint (`app/routes/saved_resumes.py`)

CSRF-exempt.

| Method | URL                              | Function               | Description                        | Auth           |
|--------|----------------------------------|------------------------|------------------------------------|----------------|
| GET    | `/api/saved-resumes`             | `list_saved_resumes()` | List user's saved resumes          | Login required |
| POST   | `/api/saved-resumes`             | `save_resume()`        | Create/update saved resume         | Login required |
| GET    | `/api/saved-resumes/<resume_id>` | `get_saved_resume()`   | Get full resume text               | Login required |
| DELETE | `/api/saved-resumes/<resume_id>` | `delete_saved_resume()`| Delete a saved resume              | Login required |

---

## Template → Route Mapping

| Template         | Served By                  | URL       |
|------------------|----------------------------|-----------|
| `index.html`     | `main_bp.index()`          | `/`       |
| `admin.html`     | `admin_bp.admin_page()`    | `/admin`  |
| `privacy.html`   | `main_bp.privacy()`        | `/privacy`|
| `terms.html`     | `main_bp.terms()`          | `/terms`  |
| `contact.html`   | `main_bp.contact()`        | `/contact`|
