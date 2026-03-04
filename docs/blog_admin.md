# Blog CMS — Admin Guide

## Overview

CVtailro includes a DB-backed blog CMS accessible from the admin dashboard. You can create, edit, publish, and delete blog posts, upload images, and manage SEO metadata — all without touching the filesystem.

## Accessing the Blog CMS

1. Go to `/admin` and log in with the admin password.
2. Click **Blog** in the sidebar under "Content."

## Creating a Post

1. Click **+ New Post**.
2. Fill in the **Title** (required).
3. Write content in **Markdown** in the editor textarea.
4. Fill in SEO fields on the right: description, keywords, category, audience.
5. Click **Save** to create a draft.
6. Click **Publish** when ready to make it live at `/blog/<slug>`.

## Editing a Post

1. In the blog list, click the post title or **Edit** button.
2. Modify any fields.
3. Click **Save**. The slug is preserved unless you change the title.
4. Use **Toggle Preview** to see a live Markdown preview.

## Publishing and Unpublishing

- **Publish**: Makes the post visible on the public `/blog` page.
- **Unpublish**: Reverts the post to draft status (hidden from public).

## Deleting a Post

1. Open the post in the editor.
2. Click **Delete** (red button). Confirm the prompt.
3. The post is permanently removed.

## Uploading Images

1. In the editor, click **Upload Image** above the preview area, or use the **Image Library** panel on the right.
2. Select an image file (JPG, PNG, GIF, WebP, SVG — max 5 MB).
3. The image is uploaded to Cloudflare R2 (if configured) or local storage.
4. A Markdown image tag `![filename](url)` is automatically inserted at the cursor position.
5. You can also click any image in the **Image Library** to insert it.

## Image Storage Configuration

Set the `R2_PUBLIC_BASE_URL` environment variable to your R2 public bucket domain (e.g., `https://cdn.yourdomain.com`). Images are stored under `blog/images/` in R2.

If R2 is not configured, images are saved locally under `static/blog-images/` and served from `/static/blog-images/`.

## Importing Existing File-Based Posts

If you have Markdown posts in `content/blog/*.md`, click **Import from Files** in the blog list view. This reads all `.md` files, parses their metadata, and inserts them into the database as published posts. Posts with duplicate slugs are skipped.

## SEO Fields

Each post supports:

- **Description**: Used as `<meta name="description">` and Open Graph description.
- **Keywords**: Comma-separated terms for `<meta name="keywords">`.
- **Category**: Used for filtering on the public blog index.
- **Audience**: `ads`, `ai`, or `general` — used for filtering and content strategy.
- **Feature Image URL**: Used for Open Graph image tags.
- **Canonical URL**: Override the default canonical URL if the post is syndicated.

## AdSense

Ad slots (top, mid, bottom) render automatically on public blog pages when `ADSENSE_CLIENT_ID` and slot IDs are configured in environment variables.
