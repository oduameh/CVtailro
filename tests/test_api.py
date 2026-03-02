"""Tests for core API endpoints."""


def test_tailor_no_resume(client):
    resp = client.post("/api/tailor", data={"job_description": "x" * 100})
    assert resp.status_code == 400
    assert "resume" in resp.get_json()["error"].lower()


def test_tailor_short_jd(client):
    import io

    data = {
        "resume": (io.BytesIO(b"test resume"), "resume.txt"),
        "job_description": "too short",
    }
    resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "short" in resp.get_json()["error"].lower()


def test_tailor_invalid_mode(client):
    import io

    data = {
        "resume": (io.BytesIO(b"test resume"), "resume.txt"),
        "job_description": "x" * 100,
        "mode": "invalid",
    }
    resp = client.post("/api/tailor", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "mode" in resp.get_json()["error"].lower()


def test_result_not_found(client):
    resp = client.get("/api/result/nonexistent123")
    assert resp.status_code == 404


def test_progress_not_found(client):
    resp = client.get("/api/progress/nonexistent123")
    assert resp.status_code == 404
