"""Shared pytest fixtures for CVtailro test suite."""

import secrets

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


def login_user_with_session(client, user):
    """Log in a user and create a valid server-side session for tests."""
    from datetime import datetime, timedelta, timezone

    from app.models.user_session import UserSession

    token = secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    sess = UserSession(
        user_id=user.id,
        session_token=token,
        ip_address="127.0.0.1",
        user_agent="TestClient/1.0",
        device_type="desktop",
        browser_name="TestBrowser",
        os_name="TestOS",
        created_at=now,
        last_activity_at=now,
        expires_at=now + timedelta(hours=24),
        is_active=True,
    )
    _db.session.add(sess)
    _db.session.commit()

    with client.session_transaction() as flask_sess:
        flask_sess["_user_id"] = user.id
        flask_sess["_fresh"] = True
        flask_sess["session_token"] = token
