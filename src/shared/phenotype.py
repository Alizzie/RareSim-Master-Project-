"""
Phenotype extraction pipeline.

Extracts HPO terms from raw clinical text using these methods below:
1. Dictionary      — exact HPO label matching (fast baseline)
2. Synonyms        — dictionary + HPO synonym expansion (handles variants)
3. Biomedical NER  — d4data transformer NER + HPO label lookup

Note: the results are not great for complex raw text - we should improve them.

Usage:
    from shared.phenotype import build_patient_profile, extract_hpo_terms

    patient, extracted_terms = build_patient_profile(
        patient_id="patient_001",
        raw_text="Patient with cerebellar ataxia and anemia.",
        hpo_labels=hpo_labels,
        methods=["dictionary", "synonyms", "biomedical_ner"],
        hpo_synonyms=hpo_synonyms,
    )
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.config import (
    BIOMEDICAL_NER_MIN_CONFIDENCE,
    BIOMEDICAL_NER_MODEL,
    HPO_BLOCKLIST,
    NEGATION_WINDOW_SIZE,
    NEGATION_WORDS,
    EXTRACTION_METHODS,
)
from shared.io import load_json, save_json
from shared.paths import (
    HPO_LABELS_PATH,
    HPO_SYNONYMS_PATH,
    OUTPUT_EXTRACTION_PATH,
    OUTPUT_PATIENT_PATH,
    PATIENT_PATH,
    PHENOTYPE_DIR,
)


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


def build_synonym_lookup(
    hpo_labels: Dict[str, str],
    hpo_synonyms: Dict[str, List[str]],
) -> Dict[str, str]:
    """Build a normalized label + synonym → HPO ID lookup."""
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


# ── Method 1: Dictionary baseline ────────────────────────────────────────────


def extract_dictionary(
    raw_text: str,
    hpo_labels: Dict[str, str],
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """Exact HPO label matching using regex."""
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
    """Dictionary matching extended with HPO ontology synonyms."""
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
    """General biomedical NER using d4data/biomedical-ner-all."""
    try:
        from transformers import pipeline
    except ImportError:
        print(
            "[phenotype] transformers not installed — skipping NER.\n"
            "Install with: pip install transformers"
        )
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
            print("[phenotype] hpo_synonyms required for synonym method — skipping.")
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


# ── Pipeline entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Run phenotype extraction on the example patient."""
    PHENOTYPE_DIR.mkdir(parents=True, exist_ok=True)

    hpo_labels = load_json(HPO_LABELS_PATH)
    hpo_synonyms = None

    if HPO_SYNONYMS_PATH.exists():
        hpo_synonyms = load_json(HPO_SYNONYMS_PATH)
    else:
        print(
            "[phenotype] hpo_synonyms.json not found — synonym method will be skipped.\n"
            "Re-run build_shared_artifacts.py to generate it."
        )

    patient_data = load_json(PATIENT_PATH)
    raw_text = patient_data.get("raw_text", "").strip()
    patient_id = patient_data.get("patient_id", "patient_001")

    print(f"Running phenotype extraction with methods: {EXTRACTION_METHODS}\n")

    patient, extracted_terms = build_patient_profile(
        patient_id=patient_id,
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=EXTRACTION_METHODS,
        hpo_synonyms=hpo_synonyms,
    )

    save_json(patient, OUTPUT_PATIENT_PATH)
    save_json(extracted_terms, OUTPUT_EXTRACTION_PATH)

    print(f"Extracted {len(extracted_terms)} HPO term(s):\n")
    for row in extracted_terms:
        print(
            f"  {row['hpo_id']} | {row['label']:<40} | "
            f"conf={row['confidence']:.2f} | "
            f"method={row['method']}"
        )

    print(f"\nPatient profile   → {OUTPUT_PATIENT_PATH}")
    print(f"Extraction detail → {OUTPUT_EXTRACTION_PATH}")


if __name__ == "__main__":
    main()
