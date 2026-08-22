"""Shared match-score formula.

Gap analysis ("before") and the pipeline's post-tailoring re-score ("after")
must use the same weighting, otherwise reported improvements compare two
different formulas and overstate gains.
"""

from __future__ import annotations


def compute_match_score(cos_sim: float, keyword_coverage: float) -> float:
    """Shared match-score formula so before/after scores are directly comparable."""
    score = round((cos_sim * 40.0) + (keyword_coverage * 40.0) + 20.0, 1)
    return max(0.0, min(100.0, score))
