"""Blog content loader and renderer."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOG_DIR = REPO_ROOT / "content" / "blog"
INCLUDE_DIR = BLOG_DIR / "_includes"


@dataclass(frozen=True)
class BlogPost:
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
    markdown_content: str
    word_count: int


def _strip_markdown(md_text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.M)
    text = re.sub(r"`{3}.*?`{3}", "", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_>#+\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_meta_body(md_text: str) -> tuple[str, str]:
    parts = md_text.split("\n\n", 1)
    if len(parts) == 2 and re.search(r"^[A-Za-z][A-Za-z0-9_-]*\s*:", parts[0], re.M):
        return parts[0], parts[1]
    return "", md_text


def _expand_includes(md_text: str) -> str:
    def _replace(match: re.Match) -> str:
        name = match.group(1).strip()
        include_path = INCLUDE_DIR / f"{name}.md"
        if not include_path.exists():
            return ""
        return include_path.read_text(encoding="utf-8")

    return re.sub(r"\[\[include:([a-zA-Z0-9_-]+)\]\]", _replace, md_text)


def _parse_meta(md_text: str) -> tuple[dict[str, str], str, str]:
    meta_block, body = _split_meta_body(md_text)
    md = markdown.Markdown(extensions=["meta", "extra", "toc"])
    html_content = md.convert(md_text)
    raw_meta = getattr(md, "Meta", {}) or {}
    meta = {k.lower(): " ".join(v).strip() for k, v in raw_meta.items()}
    return meta, html_content, body


def _reading_time_minutes(word_count: int) -> str:
    minutes = max(1, round(word_count / 200))
    return f"{minutes} min read"


def _excerpt_from_body(body: str, word_limit: int = 60) -> str:
    text = _strip_markdown(body)
    words = text.split()
    if len(words) <= word_limit:
        return " ".join(words)
    return " ".join(words[:word_limit]).strip() + "..."


def _split_html_for_mid_ad(html_content: str, min_paragraphs: int = 4) -> tuple[str, str]:
    parts = html_content.split("</p>")
    if len(parts) <= min_paragraphs:
        return html_content, ""
    intro = "</p>".join(parts[:min_paragraphs]).strip()
    if intro and not intro.endswith("</p>"):
        intro += "</p>"
    rest = "</p>".join(parts[min_paragraphs:]).strip()
    return intro, rest


def _parse_post(path: Path) -> BlogPost:
    md_text = path.read_text(encoding="utf-8")
    md_text = _expand_includes(md_text)
    meta, html_content, body = _parse_meta(md_text)
    title = meta.get("title") or path.stem.replace("-", " ").title()
    slug = meta.get("slug") or path.stem
    description = meta.get("description", "")
    keywords = [kw.strip() for kw in meta.get("keywords", "").split(",") if kw.strip()]
    category = meta.get("category", "General")
    audience = meta.get("audience", "general")
    publish_date = meta.get("publishdate", "")
    canonical = meta.get("canonical", "")
    html_intro, html_rest = _split_html_for_mid_ad(html_content)
    body_text = _strip_markdown(body)
    word_count = len(body_text.split())
    reading_time = meta.get("readingtime") or _reading_time_minutes(word_count)
    excerpt = meta.get("excerpt") or _excerpt_from_body(body)
    return BlogPost(
        slug=slug,
        title=title,
        description=description,
        keywords=keywords,
        category=category,
        audience=audience,
        publish_date=publish_date,
        reading_time=reading_time,
        canonical=canonical,
        excerpt=excerpt,
        html_content=html_content,
        html_intro=html_intro,
        html_rest=html_rest,
        markdown_content=md_text,
        word_count=word_count,
    )


def _sort_key(post: BlogPost) -> tuple:
    if post.publish_date:
        try:
            return (datetime.fromisoformat(post.publish_date), post.title.lower())
        except ValueError:
            return (datetime.min, post.title.lower())
    return (datetime.min, post.title.lower())


def _iter_post_files() -> Iterable[Path]:
    if not BLOG_DIR.exists():
        return []
    return sorted(BLOG_DIR.glob("*.md"))


@lru_cache(maxsize=1)
def _load_posts() -> list[BlogPost]:
    posts = [_parse_post(path) for path in _iter_post_files()]
    posts.sort(key=_sort_key, reverse=True)
    return posts


def list_posts(audience: str | None = None, category: str | None = None) -> list[BlogPost]:
    posts = _load_posts()
    if audience:
        posts = [post for post in posts if post.audience.lower() == audience.lower()]
    if category:
        posts = [post for post in posts if post.category.lower() == category.lower()]
    return posts


def get_post(slug: str) -> BlogPost | None:
    slug = slug.strip().lower()
    for post in _load_posts():
        if post.slug.lower() == slug:
            return post
    return None


def available_categories() -> list[str]:
    categories = {post.category for post in _load_posts()}
    return sorted(categories)


def clear_cache() -> None:
    _load_posts.cache_clear()
