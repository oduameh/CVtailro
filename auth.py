"""Backward-compatible re-exports — new code should import from app.routes.auth and app.extensions."""

from app.extensions import login_manager, oauth
from app.routes.auth import auth_bp

def init_oauth(app):
    """Legacy helper — OAuth is now initialized inside create_app()."""
    pass

__all__ = ["auth_bp", "login_manager", "init_oauth", "oauth"]
