"""
Dictionary extractor — exact HPO label matching via regex.

Fastest baseline. No model required.
Matches normalized HPO labels directly against normalized patient text.
"""

import re
from typing import Dict, List

from ._types import ExtractionMethod, ExtractionResult
from ._utils import build_label_lookup, is_negated, normalize_text


def extract_dictionary(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Exact HPO label matching using regex.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        skip_negated:  If True, skip negated mentions (e.g. "no ataxia").

    Returns:
        List of ExtractionResult, one per match found.
    """
    normalized = normalize_text(raw_text)
    lookup = build_label_lookup(hpo_labels)
    results = []

    for label_text, hpo_id in lookup.items():
        pattern = rf"\b{re.escape(label_text)}\b"
        for match in re.finditer(pattern, normalized):
            negated = is_negated(normalized, match.start())
            if skip_negated and negated:
                continue
            results.append(ExtractionResult(
                hpo_id=hpo_id,
                label=hpo_labels[hpo_id],
                matched_text=label_text,
                method=ExtractionMethod.DICTIONARY,
                confidence=1.0,
                start=match.start(),
                end=match.end(),
                negated=negated,
            ))

    return results
    