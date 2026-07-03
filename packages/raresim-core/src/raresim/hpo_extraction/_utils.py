"""
Internal utilities shared across all extraction methods.
"""

import re

from raresim.hpo_extraction._config import (
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
)

from raresim.hpo_extraction._types import ExtractionResult


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_negated(
    text: str,
    start_index: int,
    window_size: int = NEGATION_WINDOW_SIZE,
) -> bool:
    """Check whether a phenotype mention is negated."""
    before = text[max(0, start_index - window_size) : start_index]
    return any(neg in before for neg in NEGATION_WORDS)


def build_label_lookup(hpo_labels: dict[str, str]) -> dict[str, str]:
    """Build a normalized label to HPO ID lookup."""
    lookup: dict[str, str] = {}

    for hpo_id, label in hpo_labels.items():
        normalized_label = normalize_text(label)
        if normalized_label:
            lookup[normalized_label] = hpo_id

    return lookup


def deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """
    Keep the highest-confidence result per HPO ID across all methods.

    Skips structural/metadata HPO terms from HPO_BLOCKLIST.
    """
    best: dict[str, ExtractionResult] = {}

    for result in results:
        if result.hpo_id in HPO_BLOCKLIST:
            continue

        existing = best.get(result.hpo_id)
        if existing is None or result.confidence > existing.confidence:
            best[result.hpo_id] = result

    return sorted(best.values(), key=lambda result: (result.start or 0, result.hpo_id))
