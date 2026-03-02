"""Integration tests for file download flow — DB fallback, ownership."""

import pytest

from app.models import TailoringJob, User


@pytest.fixture
def user(db):
    u = User(
        google_id="test-google-123",
        email="user@example.com",
        name="Test User",
        is_admin=False,
    )
    db.session.add(u)
    db.session.commit()
    return u


def _login(client, user):
    with client.session_transaction() as sess:
        sess["_user_id"] = user.id
        sess["_fresh"] = True


@pytest.mark.integration
class TestDownloadFromDB:
    """Download endpoint — DB fallback (regenerate from markdown)."""

    def test_download_md_from_db_as_owner(self, client, user, db):
        job = TailoringJob(
            id="dljob1234567890abcd",
            user_id=user.id,
            status="completed",
            ats_resume_md="# My Resume\nContent here",
        )
        db.session.add(job)
        db.session.commit()
        _login(client, user)
        resp = client.get(f"/api/download/{job.id}/ats_resume.md")
        assert resp.status_code == 200
        assert b"# My Resume" in resp.data or "# My Resume" in resp.get_data(as_text=True)

    def test_download_denied_to_anonymous_for_user_job(self, client, user, db):
        job = TailoringJob(
            id="dljob1234567890efgh",
            user_id=user.id,
            status="completed",
            ats_resume_md="# Secret",
        )
        db.session.add(job)
        db.session.commit()
        resp = client.get(f"/api/download/{job.id}/ats_resume.md")
        assert resp.status_code == 404

    def test_download_anonymous_job_allowed(self, client, db):
        job = TailoringJob(
            id="dljob1234567890ijkl",
            user_id=None,
            status="completed",
            ats_resume_md="# Public resume",
        )
        db.session.add(job)
        db.session.commit()
        resp = client.get(f"/api/download/{job.id}/ats_resume.md")
        assert resp.status_code == 200
