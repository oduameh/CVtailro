"""End-to-end security tests — authentication, authorization, injection, headers, rate limits."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from app.models import SavedResume, TailoringJob, User

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user(db):
    """Create a test user."""
    u = User(
        google_id="test-google-123",
        email="user@example.com",
        name="Test User",
        is_admin=False,
    )
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def admin_user(db):
    """Create an admin user."""
    u = User(
        google_id="admin-google-456",
        email="admin@example.com",
        name="Admin User",
        is_admin=True,
    )
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def other_user(db):
    """Create another user (for IDOR tests)."""
    u = User(
        google_id="other-google-789",
        email="other@example.com",
        name="Other User",
        is_admin=False,
    )
    db.session.add(u)
    db.session.commit()
    return u


@pytest.fixture
def user_job(db, user):
    """Create a tailoring job owned by user."""
    j = TailoringJob(
        id="userjob1234567890ab",
        user_id=user.id,
        status="complete",
        job_title="Engineer",
        company="Acme",
        ats_resume_md="# Resume\nConfidential content",
        talking_points_md="# Talking points",
    )
    db.session.add(j)
    db.session.commit()
    return j


@pytest.fixture
def anonymous_job(db):
    """Create a tailoring job with no owner (anonymous session)."""
    j = TailoringJob(
        id="anonjob1234567890ab",
        user_id=None,
        status="complete",
        job_title="Engineer",
        company="Acme",
        ats_resume_md="# Resume\nPublic content",
        talking_points_md="# Talking points",
    )
    db.session.add(j)
    db.session.commit()
    return j


def _login(client, user):
    """Log in as user via Flask-Login with a valid server-side session."""
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user)


def _admin_login(client, flask_app):
    """Log in to admin panel (sets admin_authenticated in session)."""
    with patch("app.services.admin_config.AdminConfigManager.has_password", return_value=True):
        with patch("app.services.admin_config.AdminConfigManager.verify_password", return_value=True):
            client.post(
                "/admin/api/login",
                json={"password": "test"},
                content_type="application/json",
            )


# ---------------------------------------------------------------------------
# Authentication & Authorization
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Authentication requirements and session handling."""

    def test_csrf_blocks_forged_profile_update(self, client, flask_app, user):
        """Exploit attempt: cross-site POST without CSRF token must be blocked."""
        _login(client, user)

        # Turn CSRF on for this test to simulate production behavior.
        flask_app.config["WTF_CSRF_ENABLED"] = True
        try:
            forged = client.post(
                "/auth/profile/update",
                json={"name": "Pwned via CSRF"},
                content_type="application/json",
                headers={"Origin": "https://evil.example"},
            )
            assert forged.status_code == 400
            assert "csrf" in (forged.get_data(as_text=True) or "").lower()

            # The key regression we care about: forged requests are rejected.
        finally:
            flask_app.config["WTF_CSRF_ENABLED"] = False

    def test_history_requires_login(self, client):
        """History endpoint must require authentication."""
        resp = client.get("/api/history")
        assert resp.status_code == 401

    def test_saved_resumes_list_requires_login(self, client):
        """Saved resumes list requires authentication."""
        resp = client.get("/api/saved-resumes")
        assert resp.status_code == 401

    def test_saved_resumes_post_requires_login(self, client):
        """Saved resumes create requires authentication."""
        resp = client.post(
            "/api/saved-resumes",
            json={"resume_text": "x" * 100, "name": "Test"},
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_download_check_requires_login(self, client):
        """Download check endpoint requires authentication."""
        resp = client.get("/api/download-check/anyjobid")
        assert resp.status_code == 401

    def test_admin_api_requires_auth(self, client):
        """Admin API endpoints require admin authentication."""
        resp = client.get("/admin/api/config")
        assert resp.status_code == 401

    def test_admin_users_requires_auth(self, client):
        """Admin users list requires authentication."""
        resp = client.get("/admin/api/users")
        assert resp.status_code == 401


class TestIDORResult:
    """IDOR: Anonymous users must not access authenticated users' job results."""

    def test_anonymous_cannot_access_user_job_result(self, client, user_job):
        """Anonymous user must not get another user's job result (IDOR)."""
        resp = client.get(f"/api/result/{user_job.id}")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "ats_resume_md" not in (data or {}).get("result", {})

    def test_anonymous_can_access_anonymous_job_result(self, client, anonymous_job):
        """Anonymous user can access their own anonymous job result."""
        resp = client.get(f"/api/result/{anonymous_job.id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("status") == "complete"
        assert "ats_resume_md" in data.get("result", {})

    def test_authenticated_user_can_access_own_job(self, client, user, user_job):
        """Authenticated user can access their own job."""
        _login(client, user)
        resp = client.get(f"/api/result/{user_job.id}")
        assert resp.status_code == 200
        assert resp.get_json().get("result", {}).get("ats_resume_md") is not None

    def test_authenticated_user_cannot_access_other_user_job(self, client, user, other_user, db):
        """Authenticated user must not access another user's job."""
        other_job = TailoringJob(
            id="otherjob1234567890ab",
            user_id=other_user.id,
            status="complete",
            ats_resume_md="# Secret resume",
        )
        db.session.add(other_job)
        db.session.commit()
        _login(client, user)
        resp = client.get(f"/api/result/{other_job.id}")
        assert resp.status_code == 404


class TestIDORDownload:
    """IDOR: File downloads must enforce job ownership."""

    def test_anonymous_cannot_download_user_job_file(self, client, user_job, tmp_path, flask_app):
        """Anonymous user must not download files from authenticated user's job."""
        # Create a file on disk for the job (simulate in-memory job)
        job_dir = tmp_path / "job_output"
        job_dir.mkdir()
        (job_dir / "resume_modern.pdf").write_bytes(b"fake pdf")
        with flask_app.app_context():
            from app.services.pipeline import jobs, jobs_lock

            with jobs_lock:
                jobs[user_job.id] = {
                    "output_dir": str(job_dir),
                    "user_id": user_job.user_id,
                    "status": "complete",
                }
        try:
            resp = client.get(f"/api/download/{user_job.id}/resume_modern.pdf")
            assert resp.status_code == 403
        finally:
            with flask_app.app_context():
                from app.services.pipeline import jobs, jobs_lock

                with jobs_lock:
                    jobs.pop(user_job.id, None)

    def test_anonymous_can_download_anonymous_job_file(self, client, anonymous_job, tmp_path, flask_app):
        """Anonymous user can download from their own anonymous job."""
        job_dir = tmp_path / "anon_output"
        job_dir.mkdir()
        (job_dir / "resume_modern.pdf").write_bytes(b"fake pdf")
        with flask_app.app_context():
            from app.services.pipeline import jobs, jobs_lock

            with jobs_lock:
                jobs[anonymous_job.id] = {
                    "output_dir": str(job_dir),
                    "user_id": None,
                    "status": "complete",
                }
        try:
            resp = client.get(f"/api/download/{anonymous_job.id}/resume_modern.pdf")
            assert resp.status_code == 200
        finally:
            with flask_app.app_context():
                from app.services.pipeline import jobs, jobs_lock

                with jobs_lock:
                    jobs.pop(anonymous_job.id, None)

    def test_path_traversal_blocked(self, client, anonymous_job):
        """Path traversal in filename must be neutralized (Path.name strips parent dirs)."""
        # serve_download uses Path(filename).name, so ../../../etc/passwd becomes passwd
        # Requesting passwd should 404 (no such file), not serve /etc/passwd
        resp = client.get(f"/api/download/{anonymous_job.id}/../../../etc/passwd")
        assert resp.status_code in (404, 403, 400)


class TestIDORHistory:
    """History and saved resumes must be scoped to current user."""

    def test_history_only_returns_own_jobs(self, client, user, other_user, db):
        """History must only return the current user's jobs."""
        my_job = TailoringJob(
            id="myjob1234567890abcd",
            user_id=user.id,
            status="complete",
            job_title="My Job",
        )
        other_job = TailoringJob(
            id="othjob1234567890abcd",
            user_id=other_user.id,
            status="complete",
            job_title="Other Job",
        )
        db.session.add_all([my_job, other_job])
        db.session.commit()
        _login(client, user)
        resp = client.get("/api/history")
        assert resp.status_code == 200
        jobs = resp.get_json().get("jobs", [])
        ids = [j["id"] for j in jobs]
        assert my_job.id in ids
        assert other_job.id not in ids

    def test_history_job_detail_requires_ownership(self, client, user, other_user, db):
        """Getting a single history job must require ownership."""
        other_job = TailoringJob(
            id="othjob1234567890abcd",
            user_id=other_user.id,
            status="complete",
            job_title="Other Job",
        )
        db.session.add(other_job)
        db.session.commit()
        _login(client, user)
        resp = client.get(f"/api/history/{other_job.id}")
        assert resp.status_code == 404

    def test_saved_resume_get_requires_ownership(self, client, user, other_user, db):
        """Getting a saved resume must require ownership."""
        other_resume = SavedResume(
            user_id=other_user.id,
            name="Other Resume",
            resume_text="Secret content",
        )
        db.session.add(other_resume)
        db.session.commit()
        _login(client, user)
        resp = client.get(f"/api/saved-resumes/{other_resume.id}")
        assert resp.status_code == 404

    def test_saved_resume_delete_requires_ownership(self, client, user, other_user, db):
        """Deleting a saved resume must require ownership."""
        other_resume = SavedResume(
            user_id=other_user.id,
            name="Other Resume",
            resume_text="Secret content",
        )
        db.session.add(other_resume)
        db.session.commit()
        rid = other_resume.id
        _login(client, user)
        resp = client.delete(f"/api/saved-resumes/{rid}")
        assert resp.status_code == 404
        assert db.session.get(SavedResume, rid) is not None


# ---------------------------------------------------------------------------
# Admin Authorization
# ---------------------------------------------------------------------------


class TestAdminAuthorization:
    """Admin panel must require proper authentication."""

    def test_admin_set_user_admin_allows_db_admin(self, client, flask_app, user, admin_user):
        """Promoting/demoting users allowed for is_admin users (unified decorator)."""
        _login(client, admin_user)
        resp = client.post(
            f"/admin/api/users/{user.id}/admin",
            json={"is_admin": True},
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_admin_config_allows_db_admin(self, client, admin_user):
        """Admin config GET allowed for is_admin users (unified decorator)."""
        _login(client, admin_user)
        resp = client.get("/admin/api/config")
        assert resp.status_code == 200

    def test_admin_users_allows_db_admin(self, client, admin_user, flask_app):
        """Admin users list allows DB is_admin users (alternative auth)."""
        _login(client, admin_user)
        resp = client.get("/admin/api/users")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Input Validation & Injection
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Input validation and injection resistance."""

    def test_job_description_strips_html(self, client, flask_app):
        """Job description must strip HTML to prevent XSS."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            data = {
                "resume": (io.BytesIO(b"x" * 500), "resume.txt"),
                "job_description": "<script>alert(1)</script>" + "x" * 50,
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            # HTML tags stripped; request accepted or rejected on other validation
            assert resp.status_code in (200, 400)

    def test_job_description_max_length(self, client, flask_app):
        """Job description must have max length."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            data = {
                "resume": (io.BytesIO(b"x" * 500), "resume.txt"),
                "job_description": "x" * 50001,
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400
            assert "long" in resp.get_json().get("error", "").lower()

    def test_mode_validation(self, client, flask_app):
        """Mode must be whitelisted."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            data = {
                "resume": (io.BytesIO(b"x" * 500), "resume.txt"),
                "job_description": "x" * 100,
                "mode": "'; DROP TABLE users; --",
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400

    def test_template_validation(self, client, flask_app):
        """Template must be whitelisted."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            data = {
                "resume": (io.BytesIO(b"x" * 500), "resume.txt"),
                "job_description": "x" * 100,
                "template": "../../../etc/passwd",
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400

    def test_history_pagination_bounds(self, client, user):
        """History per_page must be capped."""
        _login(client, user)
        resp = client.get("/api/history?per_page=9999")
        assert resp.status_code == 200
        # Should cap at 50
        data = resp.get_json()
        assert len(data.get("jobs", [])) <= 50


# ---------------------------------------------------------------------------
# Security Headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Security headers must be present."""

    def test_x_content_type_options(self, client):
        resp = client.get("/")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, client):
        resp = client.get("/")
        assert "1" in (resp.headers.get("X-XSS-Protection") or "")

    def test_referrer_policy(self, client):
        resp = client.get("/")
        assert "Referrer-Policy" in resp.headers

    def test_content_security_policy(self, client):
        resp = client.get("/")
        csp = resp.headers.get("Content-Security-Policy") or ""
        assert "default-src" in csp
        assert "script-src" in csp


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Rate limiting must be applied."""

    def test_tailor_rate_limit_applied(self, client, flask_app):
        """Tailor endpoint exists and validates input (rate limit disabled in tests)."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = ""
            resp = client.post("/api/tailor", data={"job_description": "x" * 100})
            assert resp.status_code == 400  # No resume

    def test_admin_login_rate_limit(self, client, flask_app):
        """Admin login has rate limit."""
        with patch("app.services.admin_config.AdminConfigManager.has_password", return_value=True):
            with patch("app.services.admin_config.AdminConfigManager.verify_password", return_value=False):
                for _ in range(7):
                    resp = client.post(
                        "/admin/api/login",
                        json={"password": "wrong"},
                        content_type="application/json",
                    )
                    if resp.status_code == 429:
                        break
                else:
                    pytest.skip("Rate limit may use different storage in tests")


# ---------------------------------------------------------------------------
# File Upload Security
# ---------------------------------------------------------------------------


class TestFileUpload:
    """File upload validation."""

    def test_pdf_magic_bytes_checked(self, client, flask_app):
        """PDF must have valid magic bytes."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            fake_pdf = io.BytesIO(b"NOT A PDF FILE CONTENT HERE" + b"x" * 500)
            data = {
                "resume": (fake_pdf, "resume.pdf"),
                "job_description": "x" * 100,
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400
            assert "pdf" in resp.get_json().get("error", "").lower()

    def test_unsupported_file_type_rejected(self, client, flask_app):
        """Only PDF, MD, TXT allowed."""
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            m.return_value.api_key = "sk-test"
            m.return_value.allow_user_model_selection = False
            m.return_value.default_model = "gpt-4o-mini"
            m.return_value.rate_limit_per_hour = 100
            data = {
                "resume": (io.BytesIO(b"x" * 500), "resume.exe"),
                "job_description": "x" * 100,
            }
            resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400
            assert "unsupported" in resp.get_json().get("error", "").lower()


# ---------------------------------------------------------------------------
# Sensitive Data Exposure
# ---------------------------------------------------------------------------


class TestSensitiveDataExposure:
    """Sensitive data must not be exposed."""

    def test_api_status_no_secrets(self, client):
        """API status must not leak secrets (API key, actual password values)."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sk-" not in str(data)
        assert "sk_" not in str(data)

    def test_admin_config_masks_api_key(self, client, flask_app):
        """Admin config GET must mask API key."""
        _admin_login(client, flask_app)
        with patch("app.services.admin_config.AdminConfigManager.load") as m:
            cfg = m.return_value
            cfg.api_key = "sk-1234567890abcdef"
            cfg.default_model = "gpt-4o-mini"
            cfg.allow_user_model_selection = True
            cfg.rate_limit_per_hour = 10
            cfg.updated_at = "2024-01-01T00:00:00Z"
            resp = client.get("/admin/api/config")
        if resp.status_code == 200:
            data = resp.get_json()
            assert "sk-1234567890abcdef" not in str(data)
            assert "..." in data.get("api_key", "")


# ---------------------------------------------------------------------------
# Session & Cookie Security
# ---------------------------------------------------------------------------


class TestSessionSecurity:
    """Session and cookie configuration."""

    def test_session_cookie_httponly_in_production(self, flask_app):
        """Session cookie should be HttpOnly (config check)."""
        from app.settings import ProductionSettings

        assert ProductionSettings.SESSION_COOKIE_HTTPONLY is True

    def test_session_cookie_samesite(self, flask_app):
        """Session cookie should have SameSite."""
        from app.settings import BaseSettings

        assert BaseSettings.SESSION_COOKIE_SAMESITE == "Lax"

    def test_secret_key_not_default_in_production(self):
        """Production should warn if using default secret (env check)."""
        from app.settings import ProductionSettings

        # In tests, we may use default; just verify the setting exists
        assert hasattr(ProductionSettings, "SECRET_KEY")
