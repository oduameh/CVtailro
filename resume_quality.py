"""Resume Quality Analyzer — pure Python, no LLM calls.

Analyzes resume bullets for:
- Action verb strength and variety
- Quantification (metrics, numbers, percentages)
- Bullet length optimization
- Filler word detection
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# Strong action verbs (curated list)
STRONG_VERBS = {
    "achieved", "accelerated", "architected", "automated", "built", "championed",
    "consolidated", "created", "decreased", "delivered", "designed", "developed",
    "directed", "drove", "eliminated", "enabled", "engineered", "established",
    "exceeded", "expanded", "generated", "grew", "implemented", "improved",
    "increased", "initiated", "integrated", "launched", "led", "managed",
    "mentored", "migrated", "modernized", "negotiated", "optimized", "orchestrated",
    "overhauled", "pioneered", "produced", "reduced", "refactored", "resolved",
    "restructured", "revamped", "scaled", "secured", "simplified", "spearheaded",
    "streamlined", "strengthened", "surpassed", "transformed", "unified", "upgraded",
}

WEAK_VERBS = {
    "helped", "worked", "assisted", "contributed", "participated", "supported",
    "handled", "did", "made", "got", "used", "utilized", "was responsible for",
}

FILLER_WORDS = {
    "various", "multiple", "numerous", "several", "effectively", "successfully",
    "responsible for", "duties included", "tasked with", "in charge of",
    "served as", "acted as", "helped to", "worked on", "involved in",
}


@dataclass
class BulletAnalysis:
    text: str
    word_count: int
    has_metrics: bool
    action_verb: str
    verb_strength: str  # "strong", "weak", "missing"
    filler_words_found: list[str]
    score: int  # 1-10
    suggestions: list[str]


@dataclass
class ResumeQualityReport:
    total_bullets: int
    bullets_with_metrics: int
    metrics_percentage: float
    unique_verbs: int
    repeated_verbs: list[str]
    weak_verbs_used: list[str]
    filler_words_found: list[str]
    avg_bullet_length: float
    too_long_bullets: int
    too_short_bullets: int
    overall_score: int  # 1-100
    bullet_analyses: list[BulletAnalysis] = field(default_factory=list)
    improvement_summary: list[str] = field(default_factory=list)


def analyze_bullet(text: str) -> BulletAnalysis:
    """Analyze a single resume bullet point."""
    text = text.strip().lstrip("- •●○▪▸")
    words = text.split()
    word_count = len(words)

    # Check for metrics (numbers, percentages, dollar amounts)
    has_metrics = bool(re.search(r'\d+%|\$[\d,.]+|\d{2,}[+]?|\d+x\b', text))

    # Extract action verb (first word, lowercase)
    action_verb = words[0].lower().rstrip("ed,s") if words else ""
    first_word = words[0].lower() if words else ""

    # Determine verb strength
    if first_word in STRONG_VERBS or action_verb in STRONG_VERBS:
        verb_strength = "strong"
    elif first_word in WEAK_VERBS or any(text.lower().startswith(wv) for wv in WEAK_VERBS):
        verb_strength = "weak"
    else:
        verb_strength = "neutral"

    # Check for filler words
    text_lower = text.lower()
    fillers = [fw for fw in FILLER_WORDS if fw in text_lower]

    # Generate suggestions
    suggestions = []
    if not has_metrics:
        suggestions.append("Add quantifiable results (numbers, percentages, dollar amounts)")
    if verb_strength == "weak":
        suggestions.append(f"Replace weak verb '{first_word}' with a stronger action verb")
    if word_count > 35:
        suggestions.append("Shorten to under 35 words for better readability")
    if word_count < 8:
        suggestions.append("Too short — add more detail about impact and results")
    if fillers:
        suggestions.append(f"Remove filler words: {', '.join(fillers)}")

    # Score the bullet (1-10)
    score = 5
    if has_metrics:
        score += 2
    if verb_strength == "strong":
        score += 1
    elif verb_strength == "weak":
        score -= 2
    if fillers:
        score -= len(fillers)
    if 15 <= word_count <= 30:
        score += 1
    elif word_count > 35 or word_count < 8:
        score -= 1
    score = max(1, min(10, score))

    return BulletAnalysis(
        text=text,
        word_count=word_count,
        has_metrics=has_metrics,
        action_verb=first_word,
        verb_strength=verb_strength,
        filler_words_found=fillers,
        score=score,
        suggestions=suggestions,
    )


def analyze_resume(bullets: list[str]) -> ResumeQualityReport:
    """Analyze all resume bullets and generate a quality report."""
    if not bullets:
        return ResumeQualityReport(
            total_bullets=0, bullets_with_metrics=0, metrics_percentage=0,
            unique_verbs=0, repeated_verbs=[], weak_verbs_used=[],
            filler_words_found=[], avg_bullet_length=0,
            too_long_bullets=0, too_short_bullets=0, overall_score=0,
        )

    analyses = [analyze_bullet(b) for b in bullets if b.strip()]

    # Aggregate stats
    total = len(analyses)
    with_metrics = sum(1 for a in analyses if a.has_metrics)

    # Verb analysis
    all_verbs = [a.action_verb for a in analyses]
    verb_counts = {}
    for v in all_verbs:
        verb_counts[v] = verb_counts.get(v, 0) + 1
    repeated = [v for v, c in verb_counts.items() if c > 1]
    weak = list(set(a.action_verb for a in analyses if a.verb_strength == "weak"))

    # Filler words
    all_fillers = []
    for a in analyses:
        all_fillers.extend(a.filler_words_found)
    unique_fillers = list(set(all_fillers))

    # Length analysis
    lengths = [a.word_count for a in analyses]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    too_long = sum(1 for l in lengths if l > 35)
    too_short = sum(1 for l in lengths if l < 8)

    # Overall score (weighted)
    metrics_score = (with_metrics / total * 40) if total else 0
    verb_score = (len(set(all_verbs)) / total * 20) if total else 0
    no_weak_score = ((total - len(weak)) / total * 15) if total else 0
    no_filler_score = max(0, 15 - len(unique_fillers) * 3)
    length_score = max(0, 10 - too_long * 2 - too_short * 2)
    overall = int(metrics_score + verb_score + no_weak_score + no_filler_score + length_score)
    overall = max(0, min(100, overall))

    # Improvement summary
    improvements = []
    if with_metrics < total * 0.6:
        improvements.append(f"{total - with_metrics} of {total} bullets lack quantifiable metrics")
    if repeated:
        improvements.append(f"Repeated verbs: {', '.join(repeated)} — vary your action verbs")
    if weak:
        improvements.append(f"Weak verbs detected: {', '.join(weak)} — use stronger alternatives")
    if unique_fillers:
        improvements.append(f"Filler words found: {', '.join(unique_fillers)} — remove for conciseness")
    if too_long:
        improvements.append(f"{too_long} bullets exceed 35 words — shorten for impact")

    return ResumeQualityReport(
        total_bullets=total,
        bullets_with_metrics=with_metrics,
        metrics_percentage=round(with_metrics / total * 100, 1) if total else 0,
        unique_verbs=len(set(all_verbs)),
        repeated_verbs=repeated,
        weak_verbs_used=weak,
        filler_words_found=unique_fillers,
        avg_bullet_length=round(avg_length, 1),
        too_long_bullets=too_long,
        too_short_bullets=too_short,
        overall_score=overall,
        bullet_analyses=analyses,
        improvement_summary=improvements,
    )


def extract_bullets_from_markdown(md_text: str) -> list[str]:
    """Extract bullet points from resume markdown text."""
    bullets = []
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "• ")):
            bullet = stripped.lstrip("-*• ").strip()
            if len(bullet) > 5:
                bullets.append(bullet)
    return bullets
