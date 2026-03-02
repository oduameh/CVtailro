"""Tests for health check and status endpoints."""


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["database"] == "healthy"
    assert data["backend"] == "openrouter"


def test_status_endpoint(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "configured" in data
    assert "has_admin_password" in data


def test_models_endpoint(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert "default" in data
    assert len(data["models"]) > 0


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"CVtailro" in resp.data


def test_admin_page(client):
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert b"CVtailro Admin" in resp.data
