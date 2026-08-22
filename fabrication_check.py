"""Deterministic fabrication guard for optimised bullets.

Verifies that hard facts (percentages, currency amounts, multipliers, years)
present in ORIGINAL bullets survive into OPTIMISED bullets. Complements the
LLM's self-reported ``fabrication_flag`` with a check the model cannot game.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# Fact patterns applied to ORIGINAL bullet text
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
_CURRENCY_RE = re.compile(
    r"[$€£]\s?\d[\d,.]*(?:\s?(?:k|m|bn|b|million|billion)\b)?", re.IGNORECASE
)
_MULTIPLIER_RE = re.compile(r"\b\d+(?:\.\d+)?x\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Suffix -> numeric multiplier; normalisation expands suffixes to full numbers
# so "$500k" and "$500,000" are treated as the same fact.
_SUFFIX_MULT = {
    "k": 1_000,
    "m": 1_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_SUFFIX_EXPAND_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(k|m|bn|b|million|billion)\b", re.IGNORECASE)


@dataclass
class FabricationViolation:
    """One optimised bullet that dropped hard facts from its original."""

    role_index: int
    bullet_index: int
    missing_facts: list[str]
    original_text: str
    optimised_text: str


@dataclass
class FabricationReport:
    """Aggregate result of checking every optimised bullet."""

    checked: int
    violations: list[FabricationViolation] = field(default_factory=list)

    @property
    def flagged_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "flagged_count": self.flagged_count,
            "violations": [
                {
                    "role_index": v.role_index,
                    "bullet_index": v.bullet_index,
                    "missing_facts": v.missing_facts,
                    "original_text": v.original_text,
                    "optimised_text": v.optimised_text,
                }
                for v in self.violations
            ],
        }


def _normalise(text: str) -> str:
    """Lowercase, drop thousands separators and '+' signs, expand k/m/bn suffixes."""
    text = text.lower()
    text = text.replace(",", "")
    text = text.replace("+", "")

    def _expand(match: re.Match) -> str:
        value = float(match.group(1)) * _SUFFIX_MULT[match.group(2)]
        return _format_amount(value)

    return re.sub(_SUFFIX_EXPAND_RE, _expand, text)


def _parse_amount(raw: str) -> float | None:
    """Convert a currency match like '$1.5k' into its full numeric value."""
    cleaned = re.sub(r"^[$€£]\s?", "", raw.strip()).strip()
    cleaned = cleaned.replace(",", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s?(k|m|bn|b|million|billion)?", cleaned, re.IGNORECASE)
    if not match:
        return None  # unparseable facts are skipped, never flagged
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    suffix = match.group(2)
    if suffix:
        value *= _SUFFIX_MULT[suffix.lower()]
    return value


def _format_amount(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def check_bullet(original_text: str, optimised_text: str) -> list[str]:
    """Return hard facts from *original_text* missing from *optimised_text*."""
    original_norm = _normalise(original_text or "")
    optimised_norm = _normalise(optimised_text or "")
    if not original_norm or original_norm == optimised_norm:
        return []

    missing: list[str] = []

    for fact in _PERCENT_RE.findall(original_text):
        # Left guard stops "35%" passing inside "135%"
        if not re.search(r"(?<![\d.])" + re.escape(fact[:-1]) + "%", optimised_norm):
            missing.append(fact)

    for match in _CURRENCY_RE.finditer(original_text):
        amount = _parse_amount(match.group())
        if amount is None:
            continue
        needle = _format_amount(amount)
        # Compare full normalised amounts: "$500k" passes if the optimised
        # text contains "500000" (e.g. from "$500,000") and vice versa.
        if not re.search(r"(?<![\d.])" + re.escape(needle) + r"(?!\d)", optimised_norm):
            missing.append(" ".join(match.group().split()))

    for fact in _MULTIPLIER_RE.findall(original_text):
        if not re.search(r"(?<![\d.])" + re.escape(fact) + r"(?![\dx])", optimised_norm):
            missing.append(fact)

    for year in dict.fromkeys(_YEAR_RE.findall(original_text)):
        if not re.search(r"(?<!\d)" + year + r"(?!\d)", optimised_norm):
            missing.append(year)

    return missing


def check_bullets(pairs: Iterable[tuple[int, int, str, str]]) -> FabricationReport:
    """Check ``(role_index, bullet_index, original, optimised)`` tuples in bulk."""
    violations: list[FabricationViolation] = []
    checked = 0
    for role_index, bullet_index, original_text, optimised_text in pairs:
        checked += 1
        missing_facts = check_bullet(original_text, optimised_text)
        if missing_facts:
            violations.append(
                FabricationViolation(
                    role_index=role_index,
                    bullet_index=bullet_index,
                    missing_facts=missing_facts,
                    original_text=original_text,
                    optimised_text=optimised_text,
                )
            )
    return FabricationReport(checked=checked, violations=violations)
