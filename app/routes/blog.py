"""Blog routes."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from app.services.blog_content import available_categories, get_post, list_posts

blog_bp = Blueprint("blog", __name__, url_prefix="/blog")


def _ads_config() -> dict[str, str | bool]:
    config = current_app.config
    client_id = config.get("ADSENSE_CLIENT_ID", "")
    return {
        "enabled": bool(client_id),
        "client_id": client_id,
        "slot_top": config.get("ADSENSE_SLOT_TOP", ""),
        "slot_mid": config.get("ADSENSE_SLOT_MID", ""),
        "slot_bottom": config.get("ADSENSE_SLOT_BOTTOM", ""),
    }


@blog_bp.route("/")
def blog_index():
    audience = request.args.get("audience") or None
    category = request.args.get("category") or None
    posts = list_posts(audience=audience, category=category)
    categories = available_categories()
    ads_config = _ads_config()
    return render_template(
        "blog/index.html",
        posts=posts,
        categories=categories,
        active_audience=audience,
        active_category=category,
        ads_config=ads_config,
    )


@blog_bp.route("/api/posts")
def api_posts():
    """Return latest blog posts as JSON for dynamic rendering."""
    posts = list_posts()[:6]
    return jsonify(
        [
            {
                "slug": p.slug,
                "title": p.title,
                "description": p.description,
                "category": p.category,
                "reading_time": p.reading_time,
                "keywords": p.keywords[:5],
            }
            for p in posts
        ]
    )


@blog_bp.route("/<slug>")
def blog_post(slug: str):
    post = get_post(slug)
    if not post:
        abort(404)
    base_url = current_app.config.get("BLOG_BASE_URL") or request.url_root.rstrip("/")
    canonical_url = post.canonical or f"{base_url}/blog/{post.slug}"
    ads_config = _ads_config()
    return render_template(
        "blog/post.html",
        post=post,
        canonical_url=canonical_url,
        ads_config=ads_config,
    )
