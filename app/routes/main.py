"""Main routes — index page, health check, status, and model listing."""

import logging
import threading
import time

import requests as http_requests
from flask import Blueprint, jsonify, render_template, request
from sqlalchemy import text

from app.extensions import db
from app.services.admin_config import AdminConfigManager
from app.services.blog_content import list_posts
from config import DEFAULT_MODEL, DEFAULT_NIM_MODEL, NIM_MODELS, RECOMMENDED_MODELS

logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)

# ── Dynamic NIM model cache ──────────────────────────────────────────────────
_nim_cache: dict | None = None
_nim_cache_time: float = 0
_nim_cache_lock = threading.Lock()
_NIM_CACHE_TTL = 300  # 5 minutes


def _fetch_nim_models(api_key: str) -> dict[str, str] | None:
    """Fetch chat models available for this NIM account. Returns {display_name: model_id}."""
    try:
        r = http_requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        models: dict[str, str] = {}
        for m in r.json().get("data", []):
            mid = m.get("id", "")
            if not mid:
                continue
            # Only chat/instruct models (skip embedding, reranking, vision-only, etc.)
            if not any(kw in mid.lower() for kw in ("instruct", "chat", "deepseek", "kimi", "nemotron", "mixtral")):
                continue
            display = mid.split("/")[-1].replace("-", " ").replace("_", " ").title()
            models[display] = mid
        return models if models else None
    except Exception:
        logger.debug("Failed to fetch NIM models", exc_info=True)
        return None


def _get_nim_models(api_key: str) -> dict[str, str]:
    """Return NIM models with caching. Falls back to hardcoded list."""
    global _nim_cache, _nim_cache_time
    with _nim_cache_lock:
        if _nim_cache and (time.time() - _nim_cache_time) < _NIM_CACHE_TTL:
            return _nim_cache
    fetched = _fetch_nim_models(api_key) if api_key else None
    with _nim_cache_lock:
        if fetched:
            _nim_cache = fetched
            _nim_cache_time = time.time()
            return fetched
        if _nim_cache:
            return _nim_cache
    return NIM_MODELS


@main_bp.route("/")
def index():
    blog_posts = list_posts()[:4]
    return render_template("index.html", blog_posts=blog_posts)


@main_bp.route("/privacy")
def privacy():
    return render_template("privacy.html")


@main_bp.route("/terms")
def terms():
    return render_template("terms.html")


@main_bp.route("/pricing")
def pricing():
    return render_template("pricing.html")


@main_bp.route("/contact")
def contact():
    return render_template("contact.html")


@main_bp.route("/api/health")
def health():
    # Always return 200 so Railway healthcheck passes during startup.
    # DB status is informational — don't block deployment on slow DB warmup.
    db_status = "healthy"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db_status = "unhealthy"
    return jsonify(
        {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "database": db_status,
            "configured": AdminConfigManager.is_configured(),
            "backend": AdminConfigManager.load().active_provider or "openrouter",
        }
    )


@main_bp.route("/api/status")
def api_status():
    return jsonify(
        {
            "configured": AdminConfigManager.is_configured(),
            "has_admin_password": AdminConfigManager.has_password(),
        }
    )


@main_bp.route("/api/models")
def list_models():
    config = AdminConfigManager.load()
    provider = request.args.get("provider") or config.active_provider or "openrouter"
    is_nim = provider == "nim"
    if is_nim:
        models = _get_nim_models(config.nim_api_key)
    else:
        models = RECOMMENDED_MODELS
    provider_default = DEFAULT_NIM_MODEL if is_nim else DEFAULT_MODEL
    valid_ids = set(models.values())
    # If saved default is valid for this provider, use it; otherwise pick first available or provider default
    if config.default_model and config.default_model in valid_ids:
        default = config.default_model
    elif provider_default in valid_ids:
        default = provider_default
    elif valid_ids:
        default = next(iter(models.values()))
    else:
        default = provider_default
    return jsonify(
        {
            "models": [{"id": model_id, "name": display_name} for display_name, model_id in models.items()],
            "default": default,
            "provider": provider,
            "user_selectable": config.allow_user_model_selection,
        }
    )
