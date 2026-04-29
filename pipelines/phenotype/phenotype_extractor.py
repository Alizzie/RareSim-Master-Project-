import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from phenotype_config import (
    BIOMEDICAL_NER_MIN_CONFIDENCE,
    BIOMEDICAL_NER_MODEL,
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
)

"""
Phenotype extraction pipeline.

Methods:
1. Dictionary      — exact HPO label matching (fast baseline)
2. Synonyms        — dictionary + HPO synonym expansion (handles variants)
3. Biomedical NER  — d4data transformer NER + HPO label lookup

Excluded methods due to dependency issues or performance on short clinical text:
- scispaCy  : incompatible with Python 3.12 numpy/sklearn environment
- PhenoBERT : stanza mimic model no longer publicly hosted
- Embedding : false positives on short clinical text
- LLM       : hallucinations without GPU server

The extract_hpo_terms() function is the main entry point and merges
results from all requested methods.

Sequential pipeline:
    raw text → extract_hpo_terms() → HPO terms → similarity methods → diseases
"""


# ── Data types ────────────────────────────────────────────────────────────────


class ExtractionMethod(str, Enum):
    DICTIONARY = "hpo_label_dictionary_match"
    SYNONYMS = "hpo_synonym_expansion"
    BIOMEDICAL_NER = "biomedical_ner_d4data"


@dataclass
class ExtractionResult:
    hpo_id: str
    label: str
    matched_text: str
    method: ExtractionMethod
    confidence: float = 1.0
    start: Optional[int] = None
    end: Optional[int] = None
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "hpo_id": self.hpo_id,
            "label": self.label,
            "matched_text": self.matched_text,
            "method": self.method.value,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
            "negated": self.negated,
        }


# ── Shared utils ──────────────────────────────────────────────────────────────


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
    """
    Check whether a phenotype mention is negated.

    Looks at the window of text immediately before the match.
    Example: 'no ataxia' → negated=True.
    """
    before = text[max(0, start_index - window_size) : start_index]
    return any(neg in before for neg in NEGATION_WORDS)


def build_label_lookup(hpo_labels: Dict[str, str]) -> Dict[str, str]:
    """Build a normalized label → HPO ID lookup."""
    return {
        normalize_text(label): hpo_id
        for hpo_id, label in hpo_labels.items()
        if normalize_text(label)
    }


def build_synonym_lookup(
    hpo_labels: Dict[str, str],
    hpo_synonyms: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Build a normalized label + synonym → HPO ID lookup.

    Extends the primary label lookup with synonyms from the HPO ontology.
    hpo_synonyms format: { "HP:0001251": ["cerebellar ataxia", "ataxia, cerebellar", ...] }

    This allows matching clinical variants like:
    "difficulty walking"  → HP:0001288 (gait ataxia)
    "delayed milestones"  → HP:0001263 (global developmental delay)
    """
    lookup = build_label_lookup(hpo_labels)

    for hpo_id, synonyms in hpo_synonyms.items():
        for syn in synonyms:
            normalized = normalize_text(syn)
            if normalized and normalized not in lookup:
                lookup[normalized] = hpo_id

    return lookup


def deduplicate(results: List[ExtractionResult]) -> List[ExtractionResult]:
    """
    Keep the highest-confidence result per HPO ID across all methods.
    Skips structural/metadata HPO terms (e.g. mode of inheritance).
    Ties are broken by character position (earlier match wins).
    """
    best: Dict[str, ExtractionResult] = {}
    for r in results:
        if r.hpo_id in HPO_BLOCKLIST:
            continue
        existing = best.get(r.hpo_id)
        if existing is None or r.confidence > existing.confidence:
            best[r.hpo_id] = r
    return sorted(best.values(), key=lambda x: (x.start or 0, x.hpo_id))


# ── Method 1: Dictionary baseline ────────────────────────────────────────────


def extract_dictionary(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Exact HPO label matching using regex.

    Fast and deterministic. Only matches labels verbatim —
    does not handle paraphrases or abbreviations.
    Use as a baseline or in combination with other methods.
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
            results.append(
                ExtractionResult(
                    hpo_id=hpo_id,
                    label=hpo_labels[hpo_id],
                    matched_text=label_text,
                    method=ExtractionMethod.DICTIONARY,
                    confidence=1.0,
                    start=match.start(),
                    end=match.end(),
                    negated=negated,
                )
            )

    return results


# ── Method 2: Synonym expansion ───────────────────────────────────────────────


def extract_synonyms(
    raw_text: str,
    hpo_labels: Dict[str, str],
    hpo_synonyms: Dict[str, List[str]],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Dictionary matching extended with HPO ontology synonyms.

    Handles clinical variants, abbreviations, and terms
    that don't match primary HPO labels exactly. Example:
    "difficulty walking" → HP:0001288 (gait ataxia)
    "delayed milestones" → HP:0001263 (global developmental delay)

    hpo_synonyms comes from the HPO ontology (hasExactSynonym,
    hasBroadSynonym fields). Built by build_shared_artifacts.py
    and saved to outputs/shared/hpo_synonyms.json.
    """
    normalized = normalize_text(raw_text)
    lookup = build_synonym_lookup(hpo_labels, hpo_synonyms)
    results = []

    for label_text, hpo_id in lookup.items():
        pattern = rf"\b{re.escape(label_text)}\b"
        for match in re.finditer(pattern, normalized):
            negated = is_negated(normalized, match.start())
            if skip_negated and negated:
                continue
            results.append(
                ExtractionResult(
                    hpo_id=hpo_id,
                    label=hpo_labels.get(hpo_id, hpo_id),
                    matched_text=label_text,
                    method=ExtractionMethod.SYNONYMS,
                    confidence=0.95,
                    start=match.start(),
                    end=match.end(),
                    negated=negated,
                )
            )

    return results


# ── Method 3: Biomedical NER (d4data) ────────────────────────────────────────


def extract_biomedical_ner(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
    model_name: str = BIOMEDICAL_NER_MODEL,
) -> List[ExtractionResult]:
    """
    General biomedical NER using d4data/biomedical-ner-all.

    Detects biomedical entity spans (disease, symptom, body part, etc.)
    then links detected spans to HPO IDs via normalized label lookup.

    Two-stage matching:
    1. Exact match: normalized span == normalized HPO label
    2. Partial match: span is substring of an HPO label or vice versa
       (skips labels shorter than 10 chars to avoid false positives)

    Limitation: only matches spans that closely resemble an HPO label.
    Spans with no matching HPO label are skipped.

    Requires:
        pip install transformers
    """
    try:
        from transformers import pipeline
    except ImportError:
        print(
            "[phenotype_extractor] transformers not installed — skipping.\n"
            "Install with: pip install transformers"
        )
        return []

    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
    )

    entities = ner(raw_text)
    lookup = build_label_lookup(hpo_labels)
    normalized_full = normalize_text(raw_text)
    results = []

    for ent in entities:
        span_text = ent["word"].strip()
        normalized_span = normalize_text(span_text)

        # Stage 1: exact match
        hpo_id = lookup.get(normalized_span)

        # Stage 2: partial match — skip short labels to avoid false positives
        if not hpo_id:
            for label_norm, candidate_id in lookup.items():
                if len(label_norm) < 10:
                    continue
                if normalized_span in label_norm or label_norm in normalized_span:
                    hpo_id = candidate_id
                    break

        if not hpo_id:
            continue

        # skip low confidence NER detections
        if float(ent["score"]) < BIOMEDICAL_NER_MIN_CONFIDENCE:
            continue

        negated = is_negated(normalized_full, ent["start"])
        if skip_negated and negated:
            continue

        results.append(
            ExtractionResult(
                hpo_id=hpo_id,
                label=hpo_labels[hpo_id],
                matched_text=span_text,
                method=ExtractionMethod.BIOMEDICAL_NER,
                confidence=float(ent["score"]),
                start=ent["start"],
                end=ent["end"],
                negated=negated,
            )
        )

    return results


# ── Ensemble entry point ──────────────────────────────────────────────────────


def extract_hpo_terms(
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] = ("dictionary",),
    skip_negated: bool = True,
    hpo_synonyms: Optional[Dict[str, List[str]]] = None,
) -> List[ExtractionResult]:
    """
    Run one or more extraction methods and merge results.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        methods:       Subset of ["dictionary", "synonyms", "biomedical_ner"].
                       Methods run in order; results merged and deduplicated
                       keeping highest-confidence per HPO ID.
        skip_negated:  If True, skip negated mentions (e.g. "no ataxia").
        hpo_synonyms:  Required for "synonyms" method.

    Returns:
        Deduplicated list of ExtractionResult, sorted by position.
    """
    all_results: List[ExtractionResult] = []

    if "dictionary" in methods:
        all_results += extract_dictionary(raw_text, hpo_labels, skip_negated)

    if "synonyms" in methods:
        if hpo_synonyms is None:
            print("[phenotype_extractor] hpo_synonyms required for synonym method — skipping.")
        else:
            all_results += extract_synonyms(raw_text, hpo_labels, hpo_synonyms, skip_negated)

    if "biomedical_ner" in methods:
        all_results += extract_biomedical_ner(raw_text, hpo_labels, skip_negated)

    return deduplicate(all_results)


# ── Patient profile builder ───────────────────────────────────────────────────


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] = ("dictionary",),
    hpo_synonyms: Optional[Dict[str, List[str]]] = None,
) -> Tuple[dict, List[dict]]:
    """
    Build a patient profile dict from raw clinical text.

    Returns:
        patient:         Dict with patient_id, raw_text, hpo_terms, methods_used.
                         Compatible with transformer_retriever.DiseaseRetriever.rank().
        extracted_terms: List of dicts with full extraction provenance.
    """
    extracted = extract_hpo_terms(
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
        hpo_synonyms=hpo_synonyms,
    )

    patient = {
        "patient_id": patient_id,
        "raw_text": raw_text,
        "hpo_terms": sorted({r.hpo_id for r in extracted}),
        "methods_used": list(methods),
    }

    return patient, [r.to_dict() for r in extracted]
