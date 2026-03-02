"""Shared pytest fixtures for CVtailro test suite."""

import pytest

from app import create_app
from app.extensions import db as _db


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests — isolated, fast, no DB/network")
    config.addinivalue_line("markers", "integration: Integration tests — DB, API, auth flows")


@pytest.fixture(scope="session")
def flask_app():
    """Create a Flask application configured for testing."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="function")
def db(flask_app):
    """Provide a clean database for each test function."""
    with flask_app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture(scope="function")
def client(flask_app, db):
    """Flask test client with a fresh database."""
    with flask_app.test_client() as c, flask_app.app_context():
        yield c
