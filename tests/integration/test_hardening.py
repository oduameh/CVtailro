"""Integration tests for hardening fixes — download regen sources, R2 guards, tracker dates."""

import pytest

from app.models import TailoringJob, User
from app.models.job import JobFile


@pytest.fixture
def user(db):
    u = User(
        google_id="hardening-google-123",
        email="hardening@example.com",
        name="Hardening User",
        is_admin=False,
    )
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user):
    from tests.conftest import login_user_with_session

    login_user_with_session(client, user)


class FakeR2:
    """Stands in for storage.r2_storage inside file_service."""

    def __init__(self, configured=True):
        self._configured = configured
        self.uploads = []
        self.presigned = []

    @property
    def is_configured(self):
        return self._configured

    def upload_file(self, job_id, filename, file_path=None, file_data=None):
        self.uploads.append((job_id, filename, len(file_data or b"")))
        return f"jobs/{job_id}/{filename}"

    def generate_presigned_url(self, r2_key, expires_in=3600):
        self.presigned.append(r2_key)
        return f"https://r2.example.com/{r2_key}"


@pytest.mark.integration
class TestCoverLetterDocxSource:
    """Cover-letter DOCX must regenerate from cover_letter_md, not the resume."""

    def test_cover_docx_uses_cover_letter_md(self, client, user, db, monkeypatch):
        job = TailoringJob(
            id="hardjob123456789a",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
            cover_letter_md="Dear Hiring Manager, cover content",
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)

        captured = {}

        def fake_docx(source_md, path, template="modern"):
            captured["source"] = source_md

        monkeypatch.setattr("app.services.file_service.generate_resume_docx", fake_docx)
        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Cover_Letter.docx")
        assert resp.status_code == 200
        assert captured["source"] == "Dear Hiring Manager, cover content"

    def test_cover_docx_404_when_no_cover_letter(self, client, user, db):
        job = TailoringJob(
            id="hardjob123456789b",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
            cover_letter_md=None,
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)

        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Cover_Letter.docx")
        assert resp.status_code == 404
        assert "cover letter" in resp.get_json()["error"].lower()


@pytest.mark.integration
class TestVirtualJobFileRows:
    """JobFile rows with empty r2_key must skip the presign tier and regenerate."""

    def test_empty_r2_key_falls_through_to_regen(self, client, user, db, monkeypatch):
        fake_r2 = FakeR2(configured=True)
        monkeypatch.setattr("app.services.file_service.r2_storage", fake_r2)

        job = TailoringJob(
            id="hardjob123456789c",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
        )
        db.session.add(job)
        db.session.add(JobFile(job_id=job.id, filename="Engineer_Acme_Minimal.pdf", r2_key=""))
        db.session.commit()
        _login(client, user)

        def fake_pdf(source_md, path, template="modern"):
            with open(path, "wb") as f:
                f.write(b"%PDF-fake" + b"0" * 600)

        monkeypatch.setattr("app.services.file_service.generate_resume_pdf", fake_pdf)
        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Minimal.pdf")
        assert resp.status_code == 200
        assert fake_r2.presigned == []  # never tried to presign the empty key

    def test_real_r2_key_still_redirects(self, client, user, db, monkeypatch):
        fake_r2 = FakeR2(configured=True)
        monkeypatch.setattr("app.services.file_service.r2_storage", fake_r2)

        job = TailoringJob(
            id="hardjob123456789d",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
        )
        db.session.add(job)
        db.session.add(
            JobFile(
                job_id=job.id, filename="Engineer_Acme_Modern.pdf", r2_key="jobs/x/Engineer_Acme_Modern.pdf"
            )
        )
        db.session.commit()
        _login(client, user)

        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Modern.pdf")
        assert resp.status_code == 302
        assert "r2.example.com" in resp.headers["Location"]


@pytest.mark.integration
class TestWriteBackCaching:
    """First regeneration should upload to R2 and fill in the JobFile row."""

    def test_regen_writes_back_to_r2(self, client, user, db, monkeypatch):
        fake_r2 = FakeR2(configured=True)
        monkeypatch.setattr("app.services.file_service.r2_storage", fake_r2)

        job = TailoringJob(
            id="hardjob123456789e",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
        )
        db.session.add(job)
        db.session.add(JobFile(job_id=job.id, filename="Engineer_Acme_Tech.pdf", r2_key=""))
        db.session.commit()
        _login(client, user)

        def fake_pdf(source_md, path, template="modern"):
            with open(path, "wb") as f:
                f.write(b"%PDF-fake" + b"0" * 600)

        monkeypatch.setattr("app.services.file_service.generate_resume_pdf", fake_pdf)
        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Tech.pdf")
        assert resp.status_code == 200
        assert len(fake_r2.uploads) == 1

        row = JobFile.query.filter_by(job_id=job.id, filename="Engineer_Acme_Tech.pdf").first()
        assert row.r2_key == f"jobs/{job.id}/Engineer_Acme_Tech.pdf"
        assert row.size_bytes and row.size_bytes > 0

    def test_write_back_failure_does_not_break_download(self, client, user, db, monkeypatch):
        fake_r2 = FakeR2(configured=True)

        def boom(*args, **kwargs):
            raise RuntimeError("R2 down")

        fake_r2.upload_file = boom
        monkeypatch.setattr("app.services.file_service.r2_storage", fake_r2)

        job = TailoringJob(
            id="hardjob123456789f",
            user_id=user.id,
            status="complete",
            ats_resume_md="# Resume content",
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)

        def fake_pdf(source_md, path, template="modern"):
            with open(path, "wb") as f:
                f.write(b"%PDF-fake" + b"0" * 600)

        monkeypatch.setattr("app.services.file_service.generate_resume_pdf", fake_pdf)
        resp = client.get(f"/api/download/{job.id}/Engineer_Acme_Elegant.pdf")
        assert resp.status_code == 200


@pytest.mark.integration
class TestTrackerDateValidation:
    """Invalid non-empty dates must 400 instead of being silently dropped."""

    def test_create_with_invalid_date_returns_400(self, client, user, db):
        _login(client, user)
        resp = client.post(
            "/api/tracker",
            json={"company": "Acme", "job_title": "Engineer", "applied_date": "not-a-date"},
        )
        assert resp.status_code == 400
        assert "date" in resp.get_json()["error"].lower()

    def test_create_with_valid_date_works(self, client, user, db):
        _login(client, user)
        resp = client.post(
            "/api/tracker",
            json={"company": "Acme", "job_title": "Engineer", "applied_date": "2026-06-01"},
        )
        assert resp.status_code == 201

    def test_create_without_date_works(self, client, user, db):
        _login(client, user)
        resp = client.post("/api/tracker", json={"company": "Acme", "job_title": "Engineer"})
        assert resp.status_code == 201

    def test_update_with_invalid_date_returns_400(self, client, user, db):
        _login(client, user)
        created = client.post("/api/tracker", json={"company": "Acme", "job_title": "Engineer"})
        app_id = created.get_json()["id"]
        resp = client.put(f"/api/tracker/{app_id}", json={"interview_date": "garbage"})
        assert resp.status_code == 400
