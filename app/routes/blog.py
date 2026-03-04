"""Blog routes — DB-first with file fallback."""

from __future__ import annotations

from dataclasses import dataclass

from flask import Blueprint, abort, current_app, jsonify, render_template, request

blog_bp = Blueprint("blog", __name__, url_prefix="/blog")


@dataclass
class _TemplatePost:
    """Unified view object for blog templates (works for both DB and file posts)."""

    slug: str
    title: str
    description: str
    keywords: list[str]
    category: str
    audience: str
    publish_date: str
    reading_time: str
    canonical: str
    excerpt: str
    html_content: str
    html_intro: str
    html_rest: str


def _db_post_to_template(db_post) -> _TemplatePost:
    from app.services.blog_store import render_markdown

    html = db_post.content_html or render_markdown(db_post.content_md or "")
    parts = html.split("</p>")
    if len(parts) > 4:
        intro = "</p>".join(parts[:4]).strip()
        if intro and not intro.endswith("</p>"):
            intro += "</p>"
        rest = "</p>".join(parts[4:]).strip()
    else:
        intro = html
        rest = ""

    kw_list = [k.strip() for k in (db_post.keywords or "").split(",") if k.strip()]
    pub_date = ""
    if db_post.published_at:
        pub_date = db_post.published_at.strftime("%Y-%m-%d")

    return _TemplatePost(
        slug=db_post.slug,
        title=db_post.title,
        description=db_post.description or "",
        keywords=kw_list,
        category=db_post.category or "General",
        audience=db_post.audience or "general",
        publish_date=pub_date,
        reading_time=db_post.reading_time or "",
        canonical=db_post.canonical_url or "",
        excerpt=db_post.description or "",
        html_content=html,
        html_intro=intro,
        html_rest=rest,
    )


def _get_db_posts(audience=None, category=None):
    try:
        from app.services.blog_store import list_published_posts

        db_posts = list_published_posts(audience=audience, category=category)
        return [_db_post_to_template(p) for p in db_posts]
    except Exception:
        return None


def _get_db_categories():
    try:
        from app.services.blog_store import published_categories

        return published_categories()
    except Exception:
        return None


def _get_db_post(slug):
    try:
        from app.services.blog_store import get_post_by_slug

        db_post = get_post_by_slug(slug)
        if db_post and db_post.status == "published":
            return _db_post_to_template(db_post)
        return None
    except Exception:
        return None


@blog_bp.route("/")
def blog_index():
    audience = request.args.get("audience") or None
    category = request.args.get("category") or None

    posts = _get_db_posts(audience=audience, category=category)
    categories = _get_db_categories()

    if posts is None:
        from app.services.blog_content import available_categories, list_posts

        posts = list_posts(audience=audience, category=category)
        categories = available_categories()

    return render_template(
        "blog/index.html",
        posts=posts,
        categories=categories or [],
        active_audience=audience,
        active_category=category,
    )


@blog_bp.route("/api/posts")
def api_posts():
    posts = _get_db_posts()
    if posts is None:
        from app.services.blog_content import list_posts

        posts = list_posts()

    result = posts[:6]
    return jsonify(
        [
            {
                "slug": p.slug,
                "title": p.title,
                "description": p.description,
                "category": p.category,
                "reading_time": p.reading_time,
                "keywords": p.keywords[:5] if isinstance(p.keywords, list) else [],
            }
            for p in result
        ]
    )


@blog_bp.route("/<slug>")
def blog_post(slug: str):
    post = _get_db_post(slug)

    if not post:
        from app.services.blog_content import get_post

        post = get_post(slug)

    if not post:
        abort(404)

    base_url = current_app.config.get("BLOG_BASE_URL") or request.url_root.rstrip("/")
    canonical_url = post.canonical or f"{base_url}/blog/{post.slug}"
    return render_template(
        "blog/post.html",
        post=post,
        canonical_url=canonical_url,
    )
