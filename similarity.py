"""
Lightweight TF-IDF cosine similarity using only the Python standard library.

No numpy, no scikit-learn, no NLTK required. Provides resume-vs-job-description
similarity scoring and keyword frequency analysis.
"""

from __future__ import annotations

import math
import re
from collections import Counter


STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "as", "if", "not", "no", "so", "up", "out", "about", "into", "over",
        "after", "before", "between", "under", "above", "such", "each", "which",
        "who", "whom", "what", "when", "where", "how", "all", "both", "few",
        "more", "most", "other", "some", "any", "only", "very", "also", "just",
        "than", "then", "now", "here", "there", "because", "while", "although",
        "though", "since", "until", "unless", "whether", "either", "neither",
        "yet", "still", "already", "even", "well", "back", "much", "many",
        "own", "same", "too", "through", "during", "get", "got", "make",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, remove stop words."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[a-z0-9\-]*[a-z0-9])?", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total tokens."""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {word: count / total for word, count in counts.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency: log(N / df) for each term."""
    n = len(documents)
    if n == 0:
        return {}
    df: Counter[str] = Counter()
    for doc in documents:
        for term in set(doc):
            df[term] += 1
    # Smoothed IDF: log((N + 1) / (df + 1)) + 1
    # This prevents shared terms from getting IDF=0 when N=2
    return {
        term: math.log((n + 1) / (count + 1)) + 1
        for term, count in df.items()
    }


def compute_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """TF-IDF vector as a sparse dictionary."""
    tf = compute_tf(tokens)
    return {term: freq * idf.get(term, 0.0) for term, freq in tf.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts.

    Returns a float in [0.0, 1.0].
    """
    shared_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[k] * vec_b[k] for k in shared_keys)

    mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def resume_job_similarity(resume_text: str, job_text: str) -> float:
    """Compute cosine similarity between a resume and a job description.

    Uses TF-IDF vectors built from both documents.

    Args:
        resume_text: Cleaned text of the resume.
        job_text: Cleaned text of the job description.

    Returns:
        Cosine similarity score in [0.0, 1.0].
    """
    resume_tokens = tokenize(resume_text)
    job_tokens = tokenize(job_text)

    idf = compute_idf([resume_tokens, job_tokens])

    resume_vec = compute_tfidf(resume_tokens, idf)
    job_vec = compute_tfidf(job_tokens, idf)

    return cosine_similarity(resume_vec, job_vec)


def keyword_frequency_analysis(
    keywords: list[str], resume_text: str, job_text: str
) -> list[dict[str, int | str]]:
    """Count occurrences of each keyword in both texts.

    Args:
        keywords: List of keywords to count.
        resume_text: Resume text to search in.
        job_text: Job description text to search in.

    Returns:
        List of dicts with keyword, job_count, resume_count, and gap.
    """
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    results = []
    for kw in keywords:
        kw_lower = kw.lower()
        job_count = len(re.findall(r"\b" + re.escape(kw_lower) + r"\b", job_lower))
        resume_count = len(
            re.findall(r"\b" + re.escape(kw_lower) + r"\b", resume_lower)
        )
        results.append(
            {
                "keyword": kw,
                "job_count": job_count,
                "resume_count": resume_count,
                "gap": job_count - resume_count,
            }
        )

    return results
