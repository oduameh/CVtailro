"""DB-backed blog store — CRUD, rendering, slug generation, and file import."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import markdown as md_lib

from app.extensions import db
from app.models.blog import BlogPost, _slugify

logger = logging.getLogger("cvtailro.blog")

ALLOWED_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "br", "hr",
    "ul", "ol", "li",
    "a", "strong", "em", "code", "pre", "blockquote",
    "img", "figure", "figcaption",
    "table", "thead", "tbody", "tr", "th", "td",
    "div", "span", "section",
    "sup", "sub", "del", "ins",
}

ALLOWED_ATTRS = {
    "a": {"href", "title", "target", "rel"},
    "img": {"src", "alt", "title", "width", "height", "loading"},
    "td": {"align"},
    "th": {"align"},
    "code": {"class"},
    "pre": {"class"},
    "div": {"class", "id"},
    "span": {"class"},
}


def _sanitize_html(html: str) -> str:
    """Remove script/style tags and event handlers for XSS prevention."""
    html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.I)
    html = re.sub(r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']", "", html, flags=re.I)
    html = re.sub(r"\s+on\w+\s*=\s*\S+", "", html, flags=re.I)
    html = re.sub(r"javascript\s*:", "", html, flags=re.I)
    return html


def render_markdown(content_md: str) -> str:
    converter = md_lib.Markdown(extensions=["extra", "toc", "meta"])
    html = converter.convert(content_md)
    return _sanitize_html(html)


def _strip_markdown(md_text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.M)
    text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_>#+\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _reading_time(word_count: int) -> str:
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min read"


def _excerpt(md_text: str, word_limit: int = 60) -> str:
    text = _strip_markdown(md_text)
    words = text.split()
    if len(words) <= word_limit:
        return " ".join(words)
    return " ".join(words[:word_limit]).strip() + "..."


def _unique_slug(title: str, exclude_id: str | None = None) -> str:
    base = _slugify(title)
    if not base:
        base = "untitled"
    slug = base
    counter = 1
    while True:
        q = BlogPost.query.filter_by(slug=slug)
        if exclude_id:
            q = q.filter(BlogPost.id != exclude_id)
        if q.first() is None:
            return slug
        slug = f"{base}-{counter}"
        counter += 1


def create_post(
    title: str,
    content_md: str = "",
    *,
    description: str = "",
    keywords: str = "",
    category: str = "General",
    audience: str = "general",
    feature_image_url: str = "",
    canonical_url: str = "",
    author_id: str | None = None,
    status: str = "draft",
    slug: str | None = None,
    published_at: datetime | None = None,
) -> BlogPost:
    if slug:
        final_slug = slug
        existing = BlogPost.query.filter_by(slug=final_slug).first()
        if existing:
            final_slug = _unique_slug(title)
    else:
        final_slug = _unique_slug(title)

    html = render_markdown(content_md)
    word_count = len(_strip_markdown(content_md).split())
    rt = _reading_time(word_count)

    post = BlogPost(
        title=title,
        slug=final_slug,
        description=description or _excerpt(content_md),
        keywords=keywords,
        category=category,
        audience=audience,
        content_md=content_md,
        content_html=html,
        status=status,
        published_at=published_at,
        feature_image_url=feature_image_url,
        reading_time=rt,
        canonical_url=canonical_url,
        author_id=author_id,
    )
    db.session.add(post)
    db.session.commit()
    return post


def update_post(post_id: str, **fields) -> BlogPost | None:
    post = db.session.get(BlogPost, post_id)
    if not post:
        return None

    if "title" in fields and fields["title"] != post.title:
        post.title = fields["title"]
        if "slug" not in fields:
            post.slug = _unique_slug(fields["title"], exclude_id=post_id)

    if "slug" in fields:
        new_slug = _slugify(fields["slug"])
        existing = BlogPost.query.filter(BlogPost.slug == new_slug, BlogPost.id != post_id).first()
        if not existing:
            post.slug = new_slug

    simple_fields = [
        "description", "keywords", "category", "audience",
        "feature_image_url", "canonical_url", "author_id",
    ]
    for f in simple_fields:
        if f in fields:
            setattr(post, f, fields[f])

    if "content_md" in fields:
        post.content_md = fields["content_md"]
        post.content_html = render_markdown(fields["content_md"])
        word_count = len(_strip_markdown(fields["content_md"]).split())
        post.reading_time = _reading_time(word_count)
        if not post.description:
            post.description = _excerpt(fields["content_md"])

    post.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return post


def publish_post(post_id: str) -> BlogPost | None:
    post = db.session.get(BlogPost, post_id)
    if not post:
        return None
    post.status = "published"
    if not post.published_at:
        post.published_at = datetime.now(timezone.utc)
    post.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return post


def unpublish_post(post_id: str) -> BlogPost | None:
    post = db.session.get(BlogPost, post_id)
    if not post:
        return None
    post.status = "draft"
    post.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return post


def delete_post(post_id: str) -> bool:
    post = db.session.get(BlogPost, post_id)
    if not post:
        return False
    db.session.delete(post)
    db.session.commit()
    return True


def get_post_by_id(post_id: str) -> BlogPost | None:
    return db.session.get(BlogPost, post_id)


def get_post_by_slug(slug: str) -> BlogPost | None:
    return BlogPost.query.filter_by(slug=slug.strip().lower()).first()


def list_admin_posts(
    status: str | None = None,
    category: str | None = None,
    search: str | None = None,
) -> list[BlogPost]:
    q = BlogPost.query
    if status:
        q = q.filter(BlogPost.status == status)
    if category:
        q = q.filter(BlogPost.category == category)
    if search:
        term = f"%{search}%"
        q = q.filter(db.or_(BlogPost.title.ilike(term), BlogPost.description.ilike(term)))
    return q.order_by(BlogPost.updated_at.desc()).all()


def list_published_posts(
    audience: str | None = None,
    category: str | None = None,
) -> list[BlogPost]:
    q = BlogPost.query.filter_by(status="published")
    if audience:
        q = q.filter(db.func.lower(BlogPost.audience) == audience.lower())
    if category:
        q = q.filter(db.func.lower(BlogPost.category) == category.lower())
    return q.order_by(BlogPost.published_at.desc()).all()


def published_categories() -> list[str]:
    rows = (
        db.session.query(BlogPost.category)
        .filter(BlogPost.status == "published")
        .distinct()
        .all()
    )
    return sorted({r[0] for r in rows if r[0]})


def import_file_posts() -> int:
    """Import existing file-based blog posts into the database."""
    from app.services.blog_content import _expand_includes, _iter_post_files, _parse_meta

    imported = 0
    for path in _iter_post_files():
        try:
            md_text = path.read_text(encoding="utf-8")
            md_text = _expand_includes(md_text)
            meta, html_content, body = _parse_meta(md_text)

            title = meta.get("title") or path.stem.replace("-", " ").title()
            slug = meta.get("slug") or path.stem

            if BlogPost.query.filter_by(slug=slug).first():
                continue

            kw = meta.get("keywords", "")
            category = meta.get("category", "General")
            audience = meta.get("audience", "general")
            description = meta.get("description", "")
            publish_date_str = meta.get("publishdate", "")
            canonical = meta.get("canonical", "")

            pub_at = None
            if publish_date_str:
                try:
                    pub_at = datetime.fromisoformat(publish_date_str)
                    if pub_at.tzinfo is None:
                        pub_at = pub_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    pub_at = datetime.now(timezone.utc)

            create_post(
                title=title,
                content_md=md_text,
                description=description,
                keywords=kw,
                category=category,
                audience=audience,
                canonical_url=canonical,
                status="published",
                slug=slug,
                published_at=pub_at or datetime.now(timezone.utc),
            )
            imported += 1
        except Exception as e:
            logger.warning(f"Failed to import {path.name}: {e}")
    return imported
