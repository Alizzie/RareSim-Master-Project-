"""
Internal utilities shared across all extraction methods.
"""

import re
from typing import Dict, List

from raresim.hpo_extraction._config import (
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
)
from ._types import ExtractionResult


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


def build_label_lookup(hpo_labels: Dict[str, str]) -> Dict[str, str]:
    """Build a normalized label → HPO ID lookup."""
    return {
        normalize_text(label): hpo_id
        for hpo_id, label in hpo_labels.items()
        if normalize_text(label)
    }


def deduplicate(results: List[ExtractionResult]) -> List[ExtractionResult]:
    """
    Keep the highest-confidence result per HPO ID across all methods.
    Skips structural/metadata HPO terms (HPO_BLOCKLIST).
    """
    best: Dict[str, ExtractionResult] = {}
    for r in results:
        if r.hpo_id in HPO_BLOCKLIST:
            continue
        existing = best.get(r.hpo_id)
        if existing is None or r.confidence > existing.confidence:
            best[r.hpo_id] = r
    return sorted(best.values(), key=lambda x: (x.start or 0, x.hpo_id))
