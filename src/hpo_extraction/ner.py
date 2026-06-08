"""
Biomedical NER extractor — d4data/biomedical-ner-all transformer model.

Runs a HuggingFace NER pipeline on the raw text, then maps extracted
entity spans to HPO IDs via the label lookup.

Requires: pip install transformers
"""

from typing import Dict, List

from core.config import BIOMEDICAL_NER_MIN_CONFIDENCE, BIOMEDICAL_NER_MODEL

from ._types import ExtractionMethod, ExtractionResult
from ._utils import build_label_lookup, is_negated, normalize_text


def extract_biomedical_ner(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
    model_name: str = BIOMEDICAL_NER_MODEL,
) -> List[ExtractionResult]:
    """
    General biomedical NER using d4data/biomedical-ner-all.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        skip_negated:  If True, skip negated mentions.
        model_name:    HuggingFace model identifier to use.

    Returns:
        List of ExtractionResult for each matched entity span.
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("[ner] transformers not installed -- skipping biomedical_ner.")
        return []

    ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
    lookup = build_label_lookup(hpo_labels)
    normalized_full = normalize_text(raw_text)
    results = []

    for ent in ner(raw_text):
        span_text = ent["word"].strip()
        normalized_span = normalize_text(span_text)

        hpo_id = lookup.get(normalized_span)
        if not hpo_id:
            for label_norm, candidate_id in lookup.items():
                if len(label_norm) < 10:
                    continue
                if normalized_span in label_norm or label_norm in normalized_span:
                    hpo_id = candidate_id
                    break

        if not hpo_id:
            continue
        if float(ent["score"]) < BIOMEDICAL_NER_MIN_CONFIDENCE:
            continue

        negated = is_negated(normalized_full, ent["start"])
        if skip_negated and negated:
            continue

        results.append(ExtractionResult(
            hpo_id=hpo_id,
            label=hpo_labels[hpo_id],
            matched_text=span_text,
            method=ExtractionMethod.BIOMEDICAL_NER,
            confidence=float(ent["score"]),
            start=ent["start"],
            end=ent["end"],
            negated=negated,
        ))

    return results
    