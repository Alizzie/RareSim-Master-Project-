import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from phenotype_config import (
    BIOMEDICAL_NER_MODEL,
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
)

"""
Phenotype extraction pipeline.

Methods:
1. Dictionary      — exact HPO label matching (fast baseline)
2. scispaCy        — biomedical NER + HPO concept linking
3. Biomedical NER  — d4data/biomedical-ner-all, general biomedical
                     entity detection + HPO label lookup

Each method produces ExtractionResult objects with provenance tracking.
The extract_hpo_terms() function is the main entry point and merges
results from all requested methods.

"""


# ── Data types ────────────────────────────────────────────────────────────────


class ExtractionMethod(str, Enum):
    DICTIONARY = "hpo_label_dictionary_match"
    SCISPACY = "scispacy_entity_linking"
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

    Fast and deterministic. Only matches labels verbatim –
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


# ── Method 2: scispaCy + HPO entity linking ───────────────────────────────────


def extract_scispacy(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Biomedical NER with scispaCy + HPO concept linking.

    Handles paraphrases and clinical abbreviations better than
    dictionary matching. Requires installation:

        pip install scispacy
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
        pip install scispacy[linker]

    The HPO linker maps recognized entities directly to HPO IDs.
    Confidence score comes from the linker's similarity score.
    """
    try:
        import spacy
        from scispacy.linking import EntityLinker  # noqa: F401
    except ImportError:
        print(
            "[phenotype_extractor] scispaCy not installed — skipping.\n"
            "Install with: pip install scispacy scispacy[linker]"
        )
        return []

    nlp = spacy.load("en_core_sci_sm")

    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "hpo",
            },
        )

    doc = nlp(raw_text)
    normalized = normalize_text(raw_text)
    results = []

    for ent in doc.ents:
        if not ent._.kb_ents:
            continue

        hpo_id, score = ent._.kb_ents[0]

        if hpo_id not in hpo_labels:
            continue

        negated = is_negated(normalized, ent.start_char)
        if skip_negated and negated:
            continue

        results.append(
            ExtractionResult(
                hpo_id=hpo_id,
                label=hpo_labels[hpo_id],
                matched_text=ent.text,
                method=ExtractionMethod.SCISPACY,
                confidence=float(score),
                start=ent.start_char,
                end=ent.end_char,
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

    Limitation: only matches spans that closely resemble an HPO label.
    Spans with no matching HPO label are skipped. scispaCy is stronger
    for HPO linking — this method adds coverage for spans scispaCy misses.

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

        # Stage 2: partial match
        if not hpo_id:
            for label_norm, candidate_id in lookup.items():
                if normalized_span in label_norm or label_norm in normalized_span:
                    hpo_id = candidate_id
                    break

        if not hpo_id:
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
) -> List[ExtractionResult]:
    """
    Run one or more extraction methods and merge results.

    Args:
        raw_text:     Raw clinical patient text.
        hpo_labels:   Dict mapping HPO ID → label string.
        methods:      Subset of ["dictionary", "scispacy", "biomedical_ner"].
                      Methods are run in order; results are merged and
                      deduplicated keeping highest-confidence per HPO ID.
        skip_negated: If True, skip negated mentions (e.g. "no ataxia").

    Returns:
        Deduplicated list of ExtractionResult, sorted by position.
    """
    all_results: List[ExtractionResult] = []

    if "dictionary" in methods:
        all_results += extract_dictionary(raw_text, hpo_labels, skip_negated)

    if "scispacy" in methods:
        all_results += extract_scispacy(raw_text, hpo_labels, skip_negated)

    if "biomedical_ner" in methods:
        all_results += extract_biomedical_ner(raw_text, hpo_labels, skip_negated)

    return deduplicate(all_results)


# ── Patient profile builder ───────────────────────────────────────────────────


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] = ("dictionary",),
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
    )

    patient = {
        "patient_id": patient_id,
        "raw_text": raw_text,
        "hpo_terms": sorted({r.hpo_id for r in extracted}),
        "methods_used": list(methods),
    }

    return patient, [r.to_dict() for r in extracted]
