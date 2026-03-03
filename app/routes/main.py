"""Main routes — index page, health check, status, and model listing."""

from flask import Blueprint, jsonify, render_template
from sqlalchemy import text

from app.extensions import db
from app.services.admin_config import AdminConfigManager
from app.services.blog_content import list_posts
from config import DEFAULT_MODEL, RECOMMENDED_MODELS

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    blog_posts = list_posts()[:6]
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
            "backend": "openrouter",
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
    default = config.default_model if config.default_model else DEFAULT_MODEL
    return jsonify(
        {
            "models": [
                {"id": model_id, "name": display_name}
                for display_name, model_id in RECOMMENDED_MODELS.items()
            ],
            "default": default,
            "user_selectable": config.allow_user_model_selection,
        }
    )
