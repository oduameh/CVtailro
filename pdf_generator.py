"""Professional PDF resume generator with multiple templates.

Three production-grade styles modelled after top resume platforms:
- Executive: serif, conservative, for senior/management roles
- Modern: clean sans-serif with navy accent (default)
- Minimal: ultra-clean, typography-focused, zero color

Includes a smart markdown parser that handles resume-specific patterns:
contact blocks, role headers, categorized skills, horizontal rules.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import time
from pathlib import Path

import markdown as md_lib

if platform.system() == "Darwin":
    homebrew_lib = "/opt/homebrew/lib"
    if os.path.isdir(homebrew_lib):
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", homebrew_lib)

from weasyprint import HTML

logger = logging.getLogger(__name__)

# ─── EXECUTIVE: serif, conservative ──────────────────────────────────────────────

EXECUTIVE_CSS = """
@page { size: A4; margin: 20mm 20mm 18mm 20mm; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt; line-height: 1.45; color: #2c2c2c; }
.name { font-size: 24pt; font-weight: 400; color: #1a1a1a; text-align: center; margin-bottom: 3pt; letter-spacing: -0.3pt; }
.contact { text-align: center; font-size: 10pt; color: #555; margin-bottom: 18pt; line-height: 1.4; letter-spacing: 0.2pt; }
.contact .sep { color: #888; margin: 0 6pt; font-weight: 300; }
.section-heading { font-size: 13pt; font-weight: 600; color: #1a1a1a; text-transform: uppercase; letter-spacing: 1.8pt; border-bottom: 1pt solid #1a1a1a; padding-bottom: 6pt; margin-top: 16pt; margin-bottom: 10pt; page-break-after: avoid; }
.role { margin-bottom: 11pt; page-break-inside: avoid; }
.role-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2pt; page-break-after: avoid; }
.role-title { font-size: 11.5pt; font-weight: 600; color: #1a1a1a; letter-spacing: 0.3pt; }
.role-date { font-size: 10pt; color: #555; white-space: nowrap; flex-shrink: 0; margin-left: 18pt; text-align: right; }
.role-company { font-size: 10.5pt; color: #444; font-weight: 500; margin-bottom: 5pt; letter-spacing: 0.1pt; }
ul { margin-left: 18pt; margin-bottom: 3pt; padding-left: 0; page-break-inside: avoid; }
li { font-size: 11pt; margin-bottom: 4pt; line-height: 1.45; color: #2c2c2c; }
li::marker { color: #888; }
.summary { font-size: 11pt; line-height: 1.45; color: #333; margin-bottom: 3pt; }
.skills-text { font-size: 11pt; line-height: 1.45; color: #2c2c2c; margin-bottom: 2pt; }
.skills-category { font-size: 11pt; line-height: 1.45; margin-bottom: 2pt; }
.skills-cat-name { font-weight: 600; color: #1a1a1a; }
.edu { margin-bottom: 9pt; page-break-inside: avoid; }
.edu-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2pt; }
.edu-degree { font-size: 11.5pt; font-weight: 600; color: #1a1a1a; }
.edu-date { font-size: 10pt; color: #555; flex-shrink: 0; margin-left: 18pt; text-align: right; }
.edu-school { font-size: 10.5pt; color: #444; font-weight: 500; margin-bottom: 3pt; }
.cert { font-size: 11pt; margin-bottom: 5pt; line-height: 1.4; page-break-inside: avoid; }
.cert-name { font-weight: 600; color: #1a1a1a; }
.cert-meta { color: #555; }
.hr-divider { height: 0.5pt; background: #ccc; margin: 10pt 0; }
p { font-size: 11pt; margin-bottom: 3pt; line-height: 1.45; color: #2c2c2c; }
strong { font-weight: 600; color: #1a1a1a; }
em { font-style: italic; color: #333; }
a { color: #1a1a1a; text-decoration: none; }
"""

# ─── MODERN: sans-serif, navy accent ─────────────────────────────────────────────

MODERN_CSS = """
@page { size: A4; margin: 20mm 22mm 18mm 22mm; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 11pt; line-height: 1.4; color: #1a1a1a; }
.name { font-size: 23pt; font-weight: 700; color: #0a2540; text-align: center; margin-bottom: 5pt; letter-spacing: -0.2pt; }
.contact { text-align: center; font-size: 10.5pt; color: #4a4a4a; margin-bottom: 17pt; line-height: 1.5; letter-spacing: 0.1pt; }
.contact .sep { color: #a0a0a0; margin: 0 5pt; font-weight: 300; font-size: 10pt; }
.section-heading { font-size: 13pt; font-weight: 700; color: #0a2540; text-transform: uppercase; letter-spacing: 1.5pt; border-bottom: 2.5pt solid #0a2540; padding-bottom: 5pt; margin-top: 15pt; margin-bottom: 10pt; page-break-after: avoid; }
.role { margin-bottom: 11pt; page-break-inside: avoid; }
.role-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2pt; page-break-after: avoid; }
.role-title { font-size: 11.5pt; font-weight: 700; color: #0a2540; flex-grow: 1; }
.role-date { font-size: 10pt; color: #555; white-space: nowrap; flex-shrink: 0; margin-left: 16pt; text-align: right; }
.role-company { font-size: 10.5pt; color: #444; font-weight: 500; margin-bottom: 4pt; letter-spacing: 0.2pt; }
ul { margin-left: 16pt; margin-bottom: 3pt; padding-left: 0; page-break-inside: avoid; }
li { font-size: 11pt; margin-bottom: 4pt; line-height: 1.4; color: #1a1a1a; }
li::marker { color: #0a2540; font-weight: 600; }
.summary { font-size: 11pt; line-height: 1.4; color: #1a1a1a; margin-bottom: 3pt; }
.skills-text { font-size: 11pt; line-height: 1.4; color: #1a1a1a; margin-bottom: 2pt; }
.skills-category { font-size: 11pt; line-height: 1.4; margin-bottom: 2pt; }
.skills-cat-name { font-weight: 700; color: #0a2540; }
.edu { margin-bottom: 8pt; page-break-inside: avoid; }
.edu-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2pt; }
.edu-degree { font-size: 11.5pt; font-weight: 700; color: #0a2540; }
.edu-date { font-size: 10pt; color: #555; flex-shrink: 0; margin-left: 16pt; text-align: right; }
.edu-school { font-size: 10.5pt; color: #444; font-weight: 500; margin-bottom: 3pt; }
.cert { font-size: 11pt; margin-bottom: 4pt; line-height: 1.4; page-break-inside: avoid; }
.cert-name { font-weight: 700; color: #0a2540; }
.cert-meta { color: #555; }
.hr-divider { height: 1pt; background: #0a2540; margin: 12pt 0; opacity: 0.2; }
p { font-size: 11pt; margin-bottom: 3pt; line-height: 1.4; color: #1a1a1a; }
strong { font-weight: 700; color: #0a2540; }
em { font-style: italic; color: #444; }
a { color: #0a2540; text-decoration: none; }
"""

# ─── MINIMAL: zero color, typography only ────────────────────────────────────────

MINIMAL_CSS = """
@page { size: A4; margin: 22mm 24mm 20mm 24mm; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Inter", "Segoe UI", Roboto, "Helvetica Neue", sans-serif; font-size: 11pt; line-height: 1.5; color: #222; }
.name { font-size: 22pt; font-weight: 700; color: #000; text-align: center; margin-bottom: 4pt; letter-spacing: -0.4pt; }
.contact { text-align: center; font-size: 10pt; color: #666; margin-bottom: 20pt; line-height: 1.6; letter-spacing: 0.3pt; }
.contact .sep { color: #ccc; margin: 0 8pt; font-weight: 400; font-size: 9pt; }
.section-heading { font-size: 12pt; font-weight: 600; color: #000; text-transform: uppercase; letter-spacing: 2.2pt; border-bottom: none; padding-bottom: 8pt; margin-top: 18pt; margin-bottom: 11pt; page-break-after: avoid; }
.role { margin-bottom: 12pt; page-break-inside: avoid; }
.role-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3pt; page-break-after: avoid; }
.role-title { font-size: 11pt; font-weight: 700; color: #000; flex-grow: 1; letter-spacing: 0.2pt; }
.role-date { font-size: 10pt; color: #666; white-space: nowrap; flex-shrink: 0; margin-left: 20pt; text-align: right; }
.role-company { font-size: 10pt; color: #555; font-weight: 400; margin-bottom: 5pt; }
ul { margin-left: 18pt; margin-bottom: 4pt; padding-left: 0; page-break-inside: avoid; }
li { font-size: 11pt; margin-bottom: 5pt; line-height: 1.5; color: #222; }
li::marker { color: #ddd; }
.summary { font-size: 11pt; line-height: 1.5; color: #333; margin-bottom: 4pt; }
.skills-text { font-size: 11pt; line-height: 1.5; color: #222; margin-bottom: 3pt; }
.skills-category { font-size: 11pt; line-height: 1.5; margin-bottom: 3pt; }
.skills-cat-name { font-weight: 700; color: #000; }
.edu { margin-bottom: 10pt; page-break-inside: avoid; }
.edu-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3pt; }
.edu-degree { font-size: 11pt; font-weight: 700; color: #000; }
.edu-date { font-size: 10pt; color: #666; flex-shrink: 0; margin-left: 20pt; text-align: right; }
.edu-school { font-size: 10pt; color: #555; font-weight: 400; margin-bottom: 3pt; }
.cert { font-size: 11pt; margin-bottom: 5pt; line-height: 1.5; page-break-inside: avoid; }
.cert-name { font-weight: 700; color: #000; }
.cert-meta { color: #666; }
.hr-divider { height: 0.5pt; background: #e0e0e0; margin: 12pt 0; }
p { font-size: 11pt; margin-bottom: 4pt; line-height: 1.5; color: #222; }
strong { font-weight: 700; color: #000; }
em { font-style: italic; color: #555; }
a { color: #000; text-decoration: none; }
"""

TEMPLATES = {
    "executive": EXECUTIVE_CSS,
    "modern": MODERN_CSS,
    "minimal": MINIMAL_CSS,
}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_contact_block(lines: list[str]) -> tuple[str, list[str], int]:
    """Extract name and contact info from the top of the resume.

    Handles multiple formats the agents produce:
    - Plain name on first line, contact lines after
    - # Name as a heading
    - **City** | phone | email | link  (bold-pipe contact line)
    - Separate lines for each contact field
    - --- horizontal rules in the header area (skip them)
    """
    name = ""
    contact_lines = []
    first_section_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip horizontal rules in header
        if stripped in ("---", "***", "___"):
            continue
        # A ## or ### heading means we've hit a real section
        if re.match(r"^#{2,}\s+", stripped):
            first_section_idx = i
            break
        # A single # heading is the name
        if stripped.startswith("#"):
            if not name:
                name = re.sub(r"^#+\s*", "", stripped)
                continue
            else:
                first_section_idx = i
                break
        if not name:
            name = stripped
        else:
            contact_lines.append(stripped)
    if first_section_idx == 0:
        first_section_idx = len(lines)
    return name, contact_lines, first_section_idx


def _clean_contact_item(text: str) -> str:
    """Strip markdown from a contact line item."""
    # Remove bold markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # Render markdown links as plain text
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()


def _render_contact(name: str, contact_lines: list[str]) -> str:
    """Render name + contact as centred header.

    Handles both separate-line contact info and pipe-separated single lines.
    """
    html = f'<div class="name">{_escape(name)}</div>\n'
    if contact_lines:
        # Flatten: if a contact line contains pipes, split it into items
        items = []
        for line in contact_lines:
            if "|" in line:
                for part in line.split("|"):
                    cleaned = _clean_contact_item(part)
                    if cleaned:
                        items.append(cleaned)
            else:
                cleaned = _clean_contact_item(line)
                if cleaned:
                    items.append(cleaned)
        if items:
            sep = ' <span class="sep">\u00b7</span> '
            html += f'<div class="contact">{sep.join(_escape(c) for c in items)}</div>\n'
    return html


def _render_role(title: str, company: str, location: str, date: str) -> str:
    company_loc = company
    if location:
        company_loc += f", {location}"
    return (
        f'<div class="role">\n'
        f'  <div class="role-header">\n'
        f'    <span class="role-title">{_escape(title)}</span>\n'
        f'    <span class="role-date">{_escape(date)}</span>\n'
        f'  </div>\n'
        f'  <div class="role-company">{_escape(company_loc)}</div>\n'
    )


def _render_edu(degree: str, school: str, date: str) -> str:
    return (
        f'<div class="edu">\n'
        f'  <div class="edu-header">\n'
        f'    <span class="edu-degree">{_escape(degree)}</span>\n'
        f'    <span class="edu-date">{_escape(date)}</span>\n'
        f'  </div>\n'
        f'  <div class="edu-school">{_escape(school)}</div>\n'
    )


def _render_skills(text: str) -> str:
    cleaned = re.sub(r"\s*,\s*", ", ", text.strip().rstrip(","))
    return f'<p class="skills-text">{_escape(cleaned)}</p>\n'


def _inline_md(text: str) -> str:
    """Process inline markdown: bold, italic, links."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def markdown_to_html(md_content: str) -> str:
    """Convert resume markdown to structured HTML."""
    lines = md_content.split("\n")
    name, contact_lines, start_idx = _parse_contact_block(lines)
    html_parts = [_render_contact(name, contact_lines)]

    current_section = ""
    i = start_idx
    in_role = False
    in_edu = False
    list_items: list[str] = []

    def flush_list():
        nonlocal list_items
        if list_items:
            html_parts.append("<ul>")
            for item in list_items:
                html_parts.append(f"  <li>{item}</li>")
            html_parts.append("</ul>")
            list_items = []

    def close_role():
        nonlocal in_role
        if in_role:
            flush_list()
            html_parts.append("</div>")
            in_role = False

    def close_edu():
        nonlocal in_edu
        if in_edu:
            flush_list()
            html_parts.append("</div>")
            in_edu = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            close_role()
            close_edu()
            flush_list()
            html_parts.append('<div class="hr-divider"></div>')
            i += 1
            continue

        # Section heading: ## Heading (not ### which is a role header)
        section_match = re.match(r"^#{1,2}\s+(.+)", line)
        if section_match and not re.match(r"^###\s+", line):
            close_role()
            close_edu()
            flush_list()
            heading_text = section_match.group(1).strip()
            current_section = heading_text.lower()
            html_parts.append(f'<div class="section-heading">{_escape(heading_text)}</div>')
            i += 1
            continue

        # Role header: ### Title | Company | Location | Date  (recruiter format)
        role_h3 = re.match(r"^###\s+(.+?)(?:\s*\|\s*(.+?))?(?:\s*\|\s*(.+?))?(?:\s*\|\s*(.+?))?\s*$", line)
        if role_h3 and "experience" in current_section:
            close_role()
            flush_list()
            parts = [p.strip() for p in re.split(r"\s*\|\s*", re.sub(r"^###\s*", "", line)) if p.strip()]
            title = parts[0] if len(parts) > 0 else ""
            company = parts[1] if len(parts) > 1 else ""
            location = parts[2] if len(parts) > 2 else ""
            date = parts[3] if len(parts) > 3 else ""
            # If no date in the header, check next line
            if not date and i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl and not nl.startswith(("-", "*", "#")):
                    date = nl
                    i += 1
            html_parts.append(_render_role(title, company, location, date))
            in_role = True
            i += 1
            continue

        # Role header: **Title** | Company | Location  (ATS format)
        role3 = re.match(r"^\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.+)", line)
        if role3 and "experience" in current_section:
            close_role()
            flush_list()
            title, company, location = role3.group(1), role3.group(2), role3.group(3)
            date = ""
            if i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl and not nl.startswith(("-", "*", "#", "**")):
                    date = nl
                    i += 1
            html_parts.append(_render_role(title, company, location, date))
            in_role = True
            i += 1
            continue

        # Role header: **Title** | Company
        role2 = re.match(r"^\*\*(.+?)\*\*\s*\|\s*(.+)", line)
        if role2 and "experience" in current_section:
            close_role()
            flush_list()
            title, company = role2.group(1), role2.group(2)
            date = ""
            if i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl and not nl.startswith(("-", "*", "#", "**")):
                    date = nl
                    i += 1
            html_parts.append(_render_role(title, company, "", date))
            in_role = True
            i += 1
            continue

        # Education: **Degree** | School | Date  OR  **Degree** — School | Date
        edu_match = re.match(r"^\*\*(.+?)\*\*\s*[|—–-]+\s*(.+)", line)
        if edu_match and "education" in current_section:
            close_edu()
            flush_list()
            degree = edu_match.group(1)
            rest = edu_match.group(2)
            # Split remaining by | or —
            rest_parts = [p.strip() for p in re.split(r"\s*[|—–]\s*", rest) if p.strip()]
            school = rest_parts[0] if rest_parts else ""
            date = rest_parts[1] if len(rest_parts) > 1 else ""
            # Check next line for date if not found
            if not date and i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if nl and not nl.startswith(("-", "*", "#", "**")):
                    date = nl
                    i += 1
            html_parts.append(_render_edu(degree, school, date))
            in_edu = True
            i += 1
            continue

        # Certification: **Name** | meta  OR  **Name** — meta  OR  plain text with — or |
        if "certification" in current_section and stripped:
            close_role()
            close_edu()
            flush_list()
            cert_bold = re.match(r"^\*\*(.+?)\*\*\s*[|—–-]+\s*(.+)", stripped)
            cert_plain = re.match(r"^(.+?)\s*[—–]\s*(.+)", stripped) if not cert_bold else None
            if cert_bold:
                html_parts.append(
                    f'<div class="cert"><span class="cert-name">{_escape(cert_bold.group(1))}</span>'
                    f' &mdash; <span class="cert-meta">{_escape(cert_bold.group(2))}</span></div>'
                )
            elif cert_plain:
                html_parts.append(
                    f'<div class="cert"><span class="cert-name">{_escape(cert_plain.group(1))}</span>'
                    f' &mdash; <span class="cert-meta">{_escape(cert_plain.group(2))}</span></div>'
                )
            else:
                html_parts.append(f'<div class="cert"><span class="cert-name">{_escape(stripped)}</span></div>')
            i += 1
            continue

        # Bullet point
        bullet_match = re.match(r"^\s*[-*]\s+(.+)", line)
        if bullet_match:
            list_items.append(_inline_md(bullet_match.group(1)))
            i += 1
            continue

        # Skills — categorized: **Category:** items
        cat_skill = re.match(r"^\*\*(.+?)\*\*\s*:\s*(.+)", stripped)
        if cat_skill and "skill" in current_section:
            close_role()
            flush_list()
            html_parts.append(
                f'<p class="skills-category"><span class="skills-cat-name">'
                f'{_escape(cat_skill.group(1))}:</span> {_escape(cat_skill.group(2))}</p>'
            )
            i += 1
            continue

        # Skills — plain comma-separated (no bold category prefix)
        if stripped and "skill" in current_section and not re.match(r"^\*\*", stripped):
            close_role()
            flush_list()
            skill_text = stripped
            while i + 1 < len(lines):
                nl = lines[i + 1].strip()
                if not nl or nl.startswith(("#", "**")):
                    break
                i += 1
                skill_text += " " + nl
            html_parts.append(_render_skills(skill_text))
            i += 1
            continue

        # Skills — categorized line that wasn't caught above (fallback)
        if stripped and "skill" in current_section and re.match(r"^\*\*", stripped):
            close_role()
            flush_list()
            html_parts.append(f"<p class=\"skills-category\">{_inline_md(stripped)}</p>")
            i += 1
            continue

        # Summary
        if stripped and ("summary" in current_section or "profile" in current_section):
            flush_list()
            html_parts.append(f'<p class="summary">{_escape(stripped)}</p>')
            i += 1
            continue

        # Generic text
        if stripped:
            flush_list()
            html_parts.append(f"<p>{_inline_md(stripped)}</p>")

        i += 1

    close_role()
    close_edu()
    flush_list()
    return "\n".join(html_parts)


def generate_resume_pdf(
    md_content: str,
    output_path: str | Path,
    css: str | None = None,
    template: str = "modern",
) -> Path:
    """Generate a professionally formatted PDF from markdown resume content.

    Args:
        md_content: Resume content in markdown format.
        output_path: Path for the output PDF file.
        css: Optional custom CSS override.
        template: Style name — 'executive', 'modern', or 'minimal'.

    Returns:
        Path to the generated PDF file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    style = css or TEMPLATES.get(template, MODERN_CSS)
    html_body = markdown_to_html(md_content)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>{style}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    pdf_start = time.time()
    try:
        HTML(string=full_html).write_pdf(str(out))
    except Exception as e:
        logger.error(f"WeasyPrint failed for {out}: {e}")
        raise RuntimeError(f"PDF generation failed: {e}") from e
    pdf_elapsed = time.time() - pdf_start

    if not out.exists() or out.stat().st_size < 500:
        raise RuntimeError(f"PDF generation produced empty or missing file: {out}")

    logger.info(
        f"PDF generated: {out} ({out.stat().st_size} bytes, "
        f"template: {template}, took {pdf_elapsed:.1f}s)"
    )
    return out
