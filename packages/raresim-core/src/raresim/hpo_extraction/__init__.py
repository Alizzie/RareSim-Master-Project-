"""
hpo_extraction — HPO term extraction from raw clinical text.

Public API
----------
extract_hpo_terms(text, hpo_labels, methods, skip_negated)
    Run one or more extraction methods, return deduplicated results.

build_patient_profile(patient_id, raw_text, hpo_labels, methods)
    Build a full patient dict with HPO terms + propagated ancestor terms.

ExtractionMethod
    Enum of supported method names.

ExtractionResult
    Dataclass for a single extracted HPO term with full provenance.

Supported methods
-----------------
    "dictionary"      — exact HPO label matching (fast baseline, no model)
    "biomedical_ner"  — d4data transformer NER + HPO label lookup
    "fast_hpo_cr"     — FastHPOCR morphological matching (clone required)
    "chatgpt"         — GPT-4o-mini extraction (OPENAI_API_KEY required)
    "phenobrain_api"  — PhenoBrain BERT NER via public API (no key needed)

Example
-------
    from hpo_extraction import extract_hpo_terms, build_patient_profile
    from shared.io import load_json
    from shared.paths import HPO_LABELS_PATH

    hpo_labels = load_json(HPO_LABELS_PATH)

    # just extract terms
    results = extract_hpo_terms(
        raw_text="Patient with cerebellar ataxia and anemia.",
        hpo_labels=hpo_labels,
        methods=["dictionary", "chatgpt"],
    )

    # or build a full patient profile
    patient, extracted = build_patient_profile(
        patient_id="patient_001",
        raw_text="Patient with cerebellar ataxia and anemia.",
        hpo_labels=hpo_labels,
        methods=["dictionary", "chatgpt"],
    )
"""

from ._types import ExtractionMethod, ExtractionResult
from .ensemble import build_patient_profile, extract_hpo_terms

__all__ = [
    "extract_hpo_terms",
    "build_patient_profile",
    "ExtractionMethod",
    "ExtractionResult",
]
