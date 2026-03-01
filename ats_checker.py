"""ATS Formatting Checker — analyzes uploaded PDF resumes for ATS compatibility.

Uses pdfplumber to detect formatting issues that break ATS parsing:
- Multi-column layouts
- Tables
- Images/graphics
- Headers/footers
- Poor text extraction
- Excessive fonts
- Page count
- File size

Pure Python, no LLM calls. Returns a structured result with a 0-100 score
and actionable warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pdfplumber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ATSWarning:
    """A single ATS compatibility warning."""

    category: str   # "layout", "formatting", "content", "size"
    severity: str   # "high", "medium", "low"
    title: str
    description: str
    fix: str


@dataclass
class ATSCheckResult:
    """Complete result of an ATS compatibility check."""

    score: int = 100                                    # 0-100, higher = better
    warnings: list[ATSWarning] = field(default_factory=list)
    page_count: int = 0
    text_length: int = 0
    has_tables: bool = False
    has_images: bool = False
    is_multi_column: bool = False
    font_count: int = 0
    file_size_kb: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON responses)."""
        return {
            "score": self.score,
            "warnings": [
                {
                    "category": w.category,
                    "severity": w.severity,
                    "title": w.title,
                    "description": w.description,
                    "fix": w.fix,
                }
                for w in self.warnings
            ],
            "page_count": self.page_count,
            "text_length": self.text_length,
            "has_tables": self.has_tables,
            "has_images": self.has_images,
            "is_multi_column": self.is_multi_column,
            "font_count": self.font_count,
            "file_size_kb": round(self.file_size_kb, 1),
        }


# ---------------------------------------------------------------------------
# Internal detection helpers
# ---------------------------------------------------------------------------

def _detect_multi_column(page: pdfplumber.page.Page) -> bool:
    """Detect multi-column layout by analyzing character x-positions.

    Approach: bucket character x0 positions into left (<45% of page width)
    and right (>55% of page width) zones. If both zones contain a
    significant share of characters, the page is likely multi-column.
    """
    chars = page.chars
    if not chars:
        return False

    page_width = page.width
    if page_width <= 0:
        return False

    left_threshold = page_width * 0.45
    right_threshold = page_width * 0.55

    # Only consider printable characters (skip whitespace)
    printable = [c for c in chars if c.get("text", "").strip()]
    if len(printable) < 20:
        return False

    left_count = sum(1 for c in printable if c["x0"] < left_threshold)
    right_count = sum(1 for c in printable if c["x0"] > right_threshold)
    total = len(printable)

    # Both sides must have at least 20% of total characters
    return (left_count > total * 0.20) and (right_count > total * 0.20)


def _detect_header_footer(page: pdfplumber.page.Page) -> tuple[bool, bool]:
    """Detect text in header/footer margins.

    Header zone: top 5% of page height.
    Footer zone: bottom 5% of page height.

    Returns (has_header, has_footer).
    """
    chars = page.chars
    if not chars:
        return False, False

    page_height = page.height
    if page_height <= 0:
        return False, False

    header_threshold = page_height * 0.05
    footer_threshold = page_height * 0.95

    has_header = any(
        c.get("top", page_height) < header_threshold
        for c in chars
        if c.get("text", "").strip()
    )
    has_footer = any(
        c.get("top", 0) > footer_threshold
        for c in chars
        if c.get("text", "").strip()
    )

    return has_header, has_footer


def _collect_fonts(page: pdfplumber.page.Page) -> set[str]:
    """Collect unique font family names from a page's characters.

    Strips style suffixes like -Bold, -Italic so that
    ``TimesNewRoman-Bold`` and ``TimesNewRoman`` count as one family.
    """
    fonts: set[str] = set()
    for c in page.chars:
        raw = c.get("fontname", "")
        if not raw:
            continue
        # Normalise: strip common style suffixes to get base family
        base = raw.split("-")[0].split(",")[0].split("+")[-1]
        if base:
            fonts.add(base)
    return fonts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_pdf_ats_compatibility(
    pdf_path_or_stream: Union[str, Path, object],
    file_size_bytes: int = 0,
) -> ATSCheckResult:
    """Check a PDF resume for ATS compatibility issues.

    Args:
        pdf_path_or_stream: A file path (str/Path) **or** a file-like
            object (e.g. ``BytesIO``, Flask ``FileStorage.stream``).
        file_size_bytes: Known file size in bytes.  When the input is a
            stream the caller should pass this explicitly because we
            cannot ``stat()`` a stream.

    Returns:
        An ``ATSCheckResult`` with a 0-100 score and a list of warnings.
    """
    warnings: list[ATSWarning] = []
    score = 100
    page_count = 0
    total_text = ""
    has_tables = False
    has_images = False
    is_multi_column = False
    all_fonts: set[str] = set()
    has_header = False
    has_footer = False

    # ------------------------------------------------------------------
    # Parse the PDF
    # ------------------------------------------------------------------
    try:
        if isinstance(pdf_path_or_stream, (str, Path)):
            pdf = pdfplumber.open(str(pdf_path_or_stream))
            # Derive file size from disk if caller didn't supply it
            if file_size_bytes == 0:
                try:
                    file_size_bytes = Path(pdf_path_or_stream).stat().st_size
                except OSError:
                    pass
        else:
            pdf = pdfplumber.open(pdf_path_or_stream)

        with pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                # --- Text ---
                text = page.extract_text() or ""
                total_text += text

                # --- Tables ---
                try:
                    tables = page.find_tables()
                    if tables:
                        has_tables = True
                except Exception:
                    # find_tables can fail on malformed pages
                    pass

                # --- Images ---
                if page.images:
                    has_images = True

                # --- Multi-column ---
                if not is_multi_column:
                    is_multi_column = _detect_multi_column(page)

                # --- Header / Footer ---
                page_header, page_footer = _detect_header_footer(page)
                if page_header:
                    has_header = True
                if page_footer:
                    has_footer = True

                # --- Fonts ---
                all_fonts.update(_collect_fonts(page))

    except Exception as e:
        logger.warning("ATS check failed to parse PDF: %s", e)
        return ATSCheckResult(
            score=50,
            warnings=[
                ATSWarning(
                    category="content",
                    severity="high",
                    title="PDF Parsing Failed",
                    description=(
                        f"Could not fully analyze your PDF: {str(e)[:120]}"
                    ),
                    fix=(
                        "Ensure your PDF is not corrupted or password-protected. "
                        "Try re-exporting from Word or Google Docs."
                    ),
                )
            ],
        )

    font_count = len(all_fonts)
    file_size_kb = file_size_bytes / 1024 if file_size_bytes else 0.0

    # ------------------------------------------------------------------
    # Generate warnings & adjust score
    # ------------------------------------------------------------------

    # 1. Multi-column layout (most impactful ATS issue)
    if is_multi_column:
        score -= 20
        warnings.append(
            ATSWarning(
                category="layout",
                severity="high",
                title="Multi-column layout detected",
                description=(
                    "Two-column or multi-column layouts confuse most ATS "
                    "systems. Text may be read out of order, jumbling your "
                    "experience and skills."
                ),
                fix=(
                    "Your tailored resume uses a single-column layout for "
                    "maximum ATS compatibility."
                ),
            )
        )

    # 2. Tables
    if has_tables:
        score -= 15
        warnings.append(
            ATSWarning(
                category="formatting",
                severity="high",
                title="Tables detected",
                description=(
                    "Tables are poorly parsed by ATS systems. Content inside "
                    "tables may be skipped entirely or merged into a single line."
                ),
                fix=(
                    "Your tailored resume replaces tables with clean, "
                    "linear text formatting."
                ),
            )
        )

    # 3. Poor text extraction
    if page_count > 0:
        chars_per_page = len(total_text) / page_count
        if chars_per_page < 200:
            score -= 25
            warnings.append(
                ATSWarning(
                    category="content",
                    severity="high",
                    title="Poor text extraction",
                    description=(
                        "Very little text could be extracted from your PDF. "
                        "It may use images for text or have encoding issues."
                    ),
                    fix=(
                        "Ensure your resume is created from a text-based "
                        "editor (Word, Google Docs), not designed as an image."
                    ),
                )
            )

    # 4. Images / graphics
    if has_images:
        score -= 10
        warnings.append(
            ATSWarning(
                category="formatting",
                severity="medium",
                title="Images or graphics detected",
                description=(
                    "ATS systems cannot read text embedded in images, icons, "
                    "or decorative graphics. Important information may be lost."
                ),
                fix=(
                    "Your tailored resume uses pure text — no images or icons."
                ),
            )
        )

    # 5. Page count
    if page_count > 2:
        score -= 10
        warnings.append(
            ATSWarning(
                category="content",
                severity="medium",
                title=f"Resume is {page_count} pages",
                description=(
                    "Most recruiters prefer 1-2 page resumes. Longer resumes "
                    "may be truncated by ATS systems or ignored by reviewers."
                ),
                fix=(
                    "Trim older roles to 2 bullets each. Remove less relevant "
                    "experience to keep it concise."
                ),
            )
        )
    elif page_count == 1 and len(total_text) < 500:
        score -= 5
        warnings.append(
            ATSWarning(
                category="content",
                severity="low",
                title="Resume may be too short",
                description=(
                    "Very short resumes may not contain enough keywords to "
                    "pass ATS screening."
                ),
                fix=(
                    "Add more detail to your experience bullets and expand "
                    "your skills section."
                ),
            )
        )

    # 6. Headers / Footers
    if has_header or has_footer:
        score -= 5
        zones = []
        if has_header:
            zones.append("header")
        if has_footer:
            zones.append("footer")
        warnings.append(
            ATSWarning(
                category="layout",
                severity="medium",
                title=f"Text in {' and '.join(zones)} area",
                description=(
                    "Some ATS systems ignore text placed in page headers or "
                    "footers. Contact information or page numbers there may "
                    "be lost."
                ),
                fix=(
                    "Your tailored resume places all content in the main "
                    "body area, avoiding header/footer regions."
                ),
            )
        )

    # 7. Font variety
    if font_count > 6:
        score -= 5
        warnings.append(
            ATSWarning(
                category="formatting",
                severity="low",
                title=f"Too many fonts ({font_count})",
                description=(
                    "Using many different fonts can cause parsing issues in "
                    "some ATS systems and looks visually inconsistent."
                ),
                fix=(
                    "Your tailored resume uses a consistent, ATS-friendly "
                    "font throughout."
                ),
            )
        )

    # 8. File size
    if file_size_kb > 5000:
        score -= 5
        warnings.append(
            ATSWarning(
                category="size",
                severity="low",
                title=f"Large file size ({file_size_kb:.0f} KB)",
                description=(
                    "Very large PDF files may time out during ATS upload "
                    "or be rejected by email filters."
                ),
                fix=(
                    "Your tailored resume is optimized for a small file size."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Clamp score and handle clean result
    # ------------------------------------------------------------------
    score = max(0, min(100, score))

    if not warnings:
        warnings.append(
            ATSWarning(
                category="content",
                severity="low",
                title="No major ATS issues detected",
                description=(
                    "Your original resume appears to be ATS-friendly. "
                    "The tailored version further optimizes keyword matching "
                    "and phrasing."
                ),
                fix="",
            )
        )

    return ATSCheckResult(
        score=score,
        warnings=warnings,
        page_count=page_count,
        text_length=len(total_text),
        has_tables=has_tables,
        has_images=has_images,
        is_multi_column=is_multi_column,
        font_count=font_count,
        file_size_kb=file_size_kb,
    )
