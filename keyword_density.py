"""Keyword Density Analyzer — compares keyword frequency between JD and resume.

Produces a visual-ready comparison of how often each important keyword
appears in the job description vs the resume (before and after tailoring).
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class KeywordFrequency:
    keyword: str
    jd_count: int
    resume_before_count: int
    resume_after_count: int
    status: str  # "matched", "improved", "missing", "new"


@dataclass
class KeywordDensityReport:
    keywords: list[KeywordFrequency] = field(default_factory=list)
    total_jd_keywords: int = 0
    matched_before: int = 0
    matched_after: int = 0
    improvement: float = 0.0  # percentage improvement


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return re.findall(r'\b[a-zA-Z][a-zA-Z+#./-]{1,}\b', text.lower())


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    """Count occurrences of a phrase (case-insensitive) in text."""
    return len(re.findall(re.escape(phrase.lower()), text.lower()))


def analyze_keyword_density(
    job_description: str,
    original_resume: str,
    tailored_resume: str,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    tools: list[str] | None = None,
) -> KeywordDensityReport:
    """Analyze keyword density comparing JD vs original vs tailored resume.

    Args:
        job_description: Full job description text
        original_resume: Original resume text (before tailoring)
        tailored_resume: Tailored resume text (after AI optimization)
        required_skills: Explicitly required skills from job analysis
        preferred_skills: Preferred/nice-to-have skills
        tools: Specific tools/technologies mentioned
    """
    # Build keyword list from explicit skills + extracted from JD
    keywords_to_check = set()

    # Add explicit skills (exact phrases)
    for skill_list in [required_skills or [], preferred_skills or [], tools or []]:
        for skill in skill_list:
            if skill and len(skill) > 1:
                keywords_to_check.add(skill.strip())

    # If no explicit skills provided, extract from JD
    if not keywords_to_check:
        jd_words = _tokenize(job_description)
        word_counts = Counter(jd_words)
        # Take words appearing 2+ times that aren't common
        common_words = {"the", "and", "for", "are", "with", "this", "that", "will",
                       "you", "our", "your", "from", "have", "has", "been", "their",
                       "about", "would", "should", "could", "also", "more", "some",
                       "other", "than", "into", "over", "such", "what", "when",
                       "where", "which", "who", "how", "all", "each", "every",
                       "both", "few", "most", "any", "can", "may", "must", "shall",
                       "not", "only", "own", "same", "than", "too", "very",
                       "just", "but", "don", "now", "new", "way", "use", "get",
                       "one", "two", "per", "role", "team", "work", "well",
                       "ability", "experience", "strong", "knowledge", "skills",
                       "including", "working", "requirements", "qualifications",
                       "responsibilities", "position", "looking", "join", "company"}
        for word, count in word_counts.most_common(50):
            if count >= 2 and word not in common_words and len(word) > 2:
                keywords_to_check.add(word)

    # Analyze each keyword
    results = []
    for keyword in sorted(keywords_to_check, key=lambda k: k.lower()):
        jd_count = _count_phrase_occurrences(job_description, keyword)
        before_count = _count_phrase_occurrences(original_resume, keyword)
        after_count = _count_phrase_occurrences(tailored_resume, keyword)

        # Skip if not even in the JD
        if jd_count == 0 and before_count == 0 and after_count == 0:
            continue

        # Determine status
        if after_count > 0 and before_count == 0:
            status = "new"  # Added by tailoring
        elif after_count > before_count:
            status = "improved"  # Count increased
        elif after_count > 0:
            status = "matched"  # Present in both
        else:
            status = "missing"  # Still not in resume

        results.append(KeywordFrequency(
            keyword=keyword,
            jd_count=jd_count,
            resume_before_count=before_count,
            resume_after_count=after_count,
            status=status,
        ))

    # Sort: missing first (most actionable), then by JD frequency
    results.sort(key=lambda k: (
        0 if k.status == "missing" else 1 if k.status == "new" else 2 if k.status == "improved" else 3,
        -k.jd_count
    ))

    # Calculate summary stats
    total = len(results)
    matched_before = sum(1 for k in results if k.resume_before_count > 0)
    matched_after = sum(1 for k in results if k.resume_after_count > 0)
    improvement = ((matched_after - matched_before) / max(total, 1)) * 100 if total else 0

    return KeywordDensityReport(
        keywords=results,
        total_jd_keywords=total,
        matched_before=matched_before,
        matched_after=matched_after,
        improvement=round(improvement, 1),
    )
