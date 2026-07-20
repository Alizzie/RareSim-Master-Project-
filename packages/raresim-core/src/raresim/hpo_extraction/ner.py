"""
Biomedical NER extractor — d4data/biomedical-ner-all transformer model.

Runs a HuggingFace token-classification pipeline on the raw text, then maps
extracted entity spans to HPO IDs via the label lookup.

Requires:
    pip install transformers
"""

import importlib
from collections.abc import Callable
from typing import Any

from raresim.hpo_extraction._config import (
    BIOMEDICAL_NER_MIN_CONFIDENCE,
    BIOMEDICAL_NER_MODEL,
)
from raresim.hpo_extraction._types import ExtractionMethod, ExtractionResult
from raresim.hpo_extraction._utils import build_label_lookup, is_negated, normalize_text

# ── Optional HuggingFace dependency ────────────────────────────────────────────

_ner_pipeline_factory: Callable[..., Any] | None = None

try:
    _transformers_module = importlib.import_module("transformers")
except ImportError:
    pass
else:
    _pipeline_factory = getattr(_transformers_module, "pipeline", None)
    if callable(_pipeline_factory):
        _ner_pipeline_factory = _pipeline_factory


def _get_ner_pipeline(model_name: str) -> Any | None:
    """Create a HuggingFace token-classification pipeline if available."""
    if _ner_pipeline_factory is None:
        print("[ner] transformers not installed -- skipping biomedical_ner.")
        return None

    return _ner_pipeline_factory(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )


def _as_int_or_none(value: Any) -> int | None:
    """Return value as int only if it is already an int."""
    if isinstance(value, int):
        return value
    return None


def _parse_entity(
    entity: dict[str, Any],
) -> tuple[str, str, float, int | None, int | None] | None:
    """Extract normalized fields from one HuggingFace NER entity."""
    word_value = entity.get("word")
    if not isinstance(word_value, str):
        return None

    span_text = word_value.strip()
    normalized_span = normalize_text(span_text)
    if not normalized_span:
        return None

    try:
        score = float(entity.get("score", 0.0))
    except (TypeError, ValueError):
        return None

    start = _as_int_or_none(entity.get("start"))
    end = _as_int_or_none(entity.get("end"))

    return span_text, normalized_span, score, start, end


def _is_conservative_match(normalized_span: str, label_norm: str) -> bool:
    """Return whether an entity span and HPO label are close enough to match."""
    if len(label_norm) < 10:
        return False

    return normalized_span in label_norm or label_norm in normalized_span


def _find_hpo_id(
    normalized_span: str,
    lookup: dict[str, str],
) -> str | None:
    """Find an HPO ID for a normalized NER span."""
    hpo_id = lookup.get(normalized_span)
    if hpo_id:
        return hpo_id

    for label_norm, candidate_id in lookup.items():
        if _is_conservative_match(normalized_span, label_norm):
            return candidate_id

    return None


def _build_extraction_result( # pylint: disable=too-many-arguments, too-many-positional-arguments
    hpo_id: str,
    span_text: str,
    score: float,
    start: int | None,
    end: int | None,
    negated: bool,
    hpo_labels: dict[str, str],
) -> ExtractionResult:
    """Build one biomedical NER extraction result."""
    return ExtractionResult(
        hpo_id=hpo_id,
        label=hpo_labels.get(hpo_id) or hpo_id,
        matched_text=span_text,
        method=ExtractionMethod.BIOMEDICAL_NER,
        confidence=score,
        start=start,
        end=end,
        negated=negated,
    )


def extract_biomedical_ner( # pylint: disable=too-many-locals
    raw_text: str,
    hpo_labels: dict[str, str],
    skip_negated: bool = True,
    model_name: str = BIOMEDICAL_NER_MODEL,
) -> list[ExtractionResult]:
    """
    General biomedical NER using d4data/biomedical-ner-all.

    Args:
        raw_text: Raw clinical patient text.
        hpo_labels: Dict mapping HPO ID to label string.
        skip_negated: If True, skip negated mentions.
        model_name: HuggingFace model identifier to use.

    Returns:
        Extraction results for each matched entity span.
    """
    ner_pipeline = _get_ner_pipeline(model_name)
    if ner_pipeline is None:
        return []

    lookup = build_label_lookup(hpo_labels)
    normalized_full = normalize_text(raw_text)
    results: list[ExtractionResult] = []

    for entity in ner_pipeline(raw_text):
        entity_data = _parse_entity(entity)
        if entity_data is None:
            continue

        span_text, normalized_span, score, start, end = entity_data
        if score < BIOMEDICAL_NER_MIN_CONFIDENCE:
            continue

        hpo_id = _find_hpo_id(normalized_span, lookup)
        if hpo_id is None:
            continue

        negated = is_negated(normalized_full, start if start is not None else 0)
        if skip_negated and negated:
            continue

        results.append(
            _build_extraction_result(
                hpo_id=hpo_id,
                span_text=span_text,
                score=score,
                start=start,
                end=end,
                negated=negated,
                hpo_labels=hpo_labels,
            )
        )

    return results
