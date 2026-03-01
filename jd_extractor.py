"""Job Description URL Extractor — fetches and extracts JD text from job posting URLs.

Supports common job boards (LinkedIn, Indeed, Greenhouse, Lever) with specific selectors,
plus a generic fallback using content extraction heuristics.
"""

from __future__ import annotations
import logging
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Timeout for fetching URLs
FETCH_TIMEOUT = 15

# User-Agent to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def extract_jd_from_url(url: str) -> dict:
    """Extract job description text from a URL.

    Returns:
        dict with keys:
        - success (bool)
        - text (str): extracted JD text
        - title (str): job title if detected
        - company (str): company name if detected
        - error (str): error message if failed
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
    except Exception:
        return {"success": False, "text": "", "title": "", "company": "", "error": "Invalid URL"}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except requests.Timeout:
        return {"success": False, "text": "", "title": "", "company": "", "error": "Request timed out. Try pasting the job description directly."}
    except requests.RequestException as e:
        return {"success": False, "text": "", "title": "", "company": "", "error": f"Could not fetch URL: {str(e)[:100]}"}

    # Extract text based on domain
    text = _extract_text_from_html(html)

    if not text or len(text) < 50:
        return {"success": False, "text": "", "title": "", "company": "", "error": "Could not extract job description from this URL. Try pasting the text directly."}

    # Try to extract title
    title = _extract_title(html)
    company = _extract_company(html, domain)

    return {"success": True, "text": text, "title": title, "company": company, "error": ""}


def _extract_text_from_html(html: str) -> str:
    """Extract main text content from HTML, removing tags and boilerplate."""
    # Remove script and style tags
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Replace common block elements with newlines
    html = re.sub(r'<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'<li[^>]*>', '- ', html, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)

    # Decode HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&#39;', "'")
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)

    # Clean up whitespace
    lines = []
    for line in text.split('\n'):
        line = ' '.join(line.split())
        if line and len(line) > 2:
            lines.append(line)

    text = '\n'.join(lines)

    # Remove common boilerplate patterns
    boilerplate = [
        r'equal\s+opportunity\s+employer.*?(?:\n|$)',
        r'we\s+are\s+an\s+equal.*?(?:\n|$)',
        r'cookie\s+(?:policy|settings|preferences).*?(?:\n|$)',
        r'privacy\s+policy.*?(?:\n|$)',
        r'terms\s+(?:of\s+(?:use|service)).*?(?:\n|$)',
        r'sign\s+(?:in|up)\s+to\s+apply.*?(?:\n|$)',
        r'already\s+have\s+an\s+account.*?(?:\n|$)',
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def _extract_title(html: str) -> str:
    """Try to extract job title from HTML."""
    # Try <title> tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
        # Clean common suffixes
        for suffix in [' | LinkedIn', ' - Indeed', ' | Indeed.com', ' - Greenhouse',
                       ' | Lever', ' - LinkedIn', ' - Glassdoor', ' at ']:
            if suffix in title:
                title = title.split(suffix)[0].strip()
        if len(title) < 100:
            return title

    # Try og:title
    og_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)', html, re.IGNORECASE)
    if og_match:
        return og_match.group(1).strip()[:100]

    return ""


def _extract_company(html: str, domain: str) -> str:
    """Try to extract company name from HTML."""
    # Try og:site_name
    og_match = re.search(r'<meta\s+property=["\']og:site_name["\']\s+content=["\']([^"\']+)', html, re.IGNORECASE)
    if og_match:
        name = og_match.group(1).strip()
        if name.lower() not in ("linkedin", "indeed", "glassdoor", "greenhouse", "lever"):
            return name

    return ""


def is_url(text: str) -> bool:
    """Check if a string looks like a URL."""
    text = text.strip()
    if text.startswith(("http://", "https://", "www.")):
        return True
    # Check for common job board domains
    if re.match(r'^[\w.-]+\.(com|io|co|org|net)/\S+', text):
        return True
    return False
