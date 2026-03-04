"""BlogPost and BlogImage models — DB-backed editorial CMS."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from app.extensions import db


def _uuid():
    return uuid.uuid4().hex


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:200]


class BlogPost(db.Model):
    __tablename__ = "blog_posts"
    __table_args__ = (
        db.Index("idx_blog_posts_slug", "slug"),
        db.Index("idx_blog_posts_status", "status"),
        db.Index("idx_blog_posts_published", "status", "published_at"),
    )

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    title = db.Column(db.String(500), nullable=False)
    slug = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True, default="")
    keywords = db.Column(db.Text, nullable=True, default="")
    category = db.Column(db.String(100), nullable=True, default="General")
    audience = db.Column(db.String(50), nullable=True, default="general")
    content_md = db.Column(db.Text, nullable=False, default="")
    content_html = db.Column(db.Text, nullable=True, default="")
    status = db.Column(db.String(20), nullable=False, default="draft")
    published_at = db.Column(db.DateTime(timezone=True), nullable=True)
    feature_image_url = db.Column(db.String(1024), nullable=True, default="")
    reading_time = db.Column(db.String(50), nullable=True, default="")
    canonical_url = db.Column(db.String(1024), nullable=True, default="")
    author_id = db.Column(db.String(32), db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    author = db.relationship("User", backref=db.backref("blog_posts", lazy="dynamic"))

    def to_dict(self, include_content: bool = False) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "keywords": self.keywords,
            "category": self.category,
            "audience": self.audience,
            "status": self.status,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "feature_image_url": self.feature_image_url,
            "reading_time": self.reading_time,
            "canonical_url": self.canonical_url,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_content:
            d["content_md"] = self.content_md
            d["content_html"] = self.content_html
        return d


class BlogImage(db.Model):
    __tablename__ = "blog_images"

    id = db.Column(db.String(32), primary_key=True, default=_uuid)
    filename = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(1024), nullable=False)
    r2_key = db.Column(db.String(1024), nullable=True)
    size_bytes = db.Column(db.Integer, nullable=True)
    content_type = db.Column(db.String(100), nullable=True)
    uploaded_by = db.Column(db.String(32), db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    uploader = db.relationship("User", backref=db.backref("blog_images", lazy="dynamic"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
