"""Backward-compatible re-export — new code should import from app.services.admin_config."""

from app.services.admin_config import AdminConfig, AdminConfigManager

__all__ = ["AdminConfig", "AdminConfigManager"]
