"""Shared utilities for CVtailro: file I/O, logging setup, PDF extraction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging with console and optional file handlers.

    Args:
        verbose: If True, console shows DEBUG; otherwise INFO.
        log_file: If provided, writes DEBUG-level logs to this file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Only add console handlers once (first call). Subsequent calls from
    # pipeline threads should only add their per-job file handler.
    if not root_logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        console.setFormatter(console_fmt)
        root_logger.addHandler(console)

    # Per-job file handler (added without clearing existing handlers)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)


def extract_pdf_text(path: str | Path) -> str:
    """Extract text content from a PDF file.

    Uses pdfplumber for accurate text extraction that preserves
    layout structure, bullet points, and section ordering.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {p}")

    logger.info(f"Extracting text from PDF: {p}")
    pages_text = []
    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    if not pages_text:
        raise ValueError(
            f"No text could be extracted from {p}. "
            f"The PDF may be image-based (scanned). "
            f"Please provide a text-based PDF or a markdown file."
        )

    result = "\n\n".join(pages_text)
    logger.info(f"Extracted {len(pages_text)} pages, {len(result)} chars from {p.name}")
    return result


def load_file(path: str | Path) -> str:
    """Read a text file and return its contents.

    Args:
        path: Path to the file.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def load_resume(path: str | Path) -> str:
    """Load a resume from either a PDF or text/markdown file.

    Automatically detects the file type and uses the appropriate
    extraction method.

    Args:
        path: Path to the resume file (.pdf, .md, or .txt).

    Returns:
        Resume text content.
    """
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return extract_pdf_text(p)
    return load_file(p)


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> Path:
    """Write a dictionary or list to a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.

    Returns:
        The path that was written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
    p.write_text(content, encoding="utf-8")
    logger.debug(f"Saved JSON ({len(content)} chars) to {p}")
    return p


def save_markdown(text: str, path: str | Path) -> Path:
    """Write markdown text to a file.

    Args:
        text: Markdown content.
        path: Output file path.

    Returns:
        The path that was written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = text.strip() + "\n"
    p.write_text(content, encoding="utf-8")
    logger.debug(f"Saved markdown ({len(content)} chars) to {p}")
    return p


def create_output_dir(base: str | None = None) -> Path:
    """Create a timestamped output directory.

    Args:
        base: If provided, use this path directly. Otherwise, create
              a new timestamped subdirectory under ``output/``.

    Returns:
        Path to the output directory.
    """
    if base:
        out = Path(base)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "output" / timestamp

    out.mkdir(parents=True, exist_ok=True)
    return out
