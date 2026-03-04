"""Integration tests for health check and status endpoints."""

import pytest


@pytest.mark.integration
def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["database"] == "healthy"
    assert data["backend"] == "openrouter"


@pytest.mark.integration
def test_status_endpoint(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "configured" in data
    assert "has_admin_password" in data


@pytest.mark.integration
def test_models_endpoint(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert "default" in data
    assert len(data["models"]) > 0


@pytest.mark.integration
def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"CVtailro" in resp.data


@pytest.mark.integration
def test_admin_page(client):
    resp = client.get("/admin")
    assert resp.status_code == 200
    assert b"CVtailro Admin" in resp.data


@pytest.mark.integration
def test_robots_txt(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "User-agent:" in text
    assert "Sitemap:" in text


@pytest.mark.integration
def test_sitemap_xml(client):
    resp = client.get("/sitemap.xml")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "<urlset" in text
    assert "<loc>" in text
