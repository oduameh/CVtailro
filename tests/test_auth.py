"""Tests for authentication endpoints."""


def test_auth_me_unauthenticated(client):
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["authenticated"] is False


def test_logout_unauthenticated(client):
    resp = client.post("/auth/logout")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True


def test_history_requires_auth(client):
    resp = client.get("/api/history")
    assert resp.status_code == 401
