"""
Ensemble entry point — run one or more extraction methods and merge results.

extract_hpo_terms    : run selected methods, deduplicate, return results.
build_patient_profile: build a full patient dict with HPO terms + propagation.
"""

from typing import Dict, List, Tuple

from raresim.utils.io import load_json
from raresim.utils.hpo_utils import get_ancestors_inclusive, preprocess_ancestor_sets
from raresim.utils.paths import HPO_ANCESTORS_PATH

from ._types import ExtractionResult
from ._utils import deduplicate
from .dictionary import extract_dictionary
from .fast_hpo_cr import extract_fast_hpo_cr
from .gpt import extract_chatgpt
from .ner import extract_biomedical_ner
from .phenobrain import extract_phenobrain_api


def extract_hpo_terms(
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] | None = None,
    skip_negated: bool = True,
) -> List[ExtractionResult]:
    """
    Run one or more extraction methods and merge results.

    Args:
        raw_text:      Raw clinical patient text.
        hpo_labels:    Dict mapping HPO ID → label string.
        methods:       One or more of:
                         "dictionary"     — exact label matching (fast baseline)
                         "biomedical_ner" — d4data transformer NER
                         "fast_hpo_cr"    — FastHPOCR morphological matching
                         "chatgpt"        — GPT-4o-mini extraction
                         "phenobrain_api" — PhenoBrain public API
        skip_negated:  If True, skip negated mentions (e.g. "no ataxia").

    Returns:
        Deduplicated list of ExtractionResult, sorted by position.
    """
    all_results: List[ExtractionResult] = []
    if methods is None:
        methods = ["dictionary"]

    if "dictionary" in methods:
        all_results += extract_dictionary(raw_text, hpo_labels, skip_negated)

    if "biomedical_ner" in methods:
        all_results += extract_biomedical_ner(raw_text, hpo_labels, skip_negated)

    if "fast_hpo_cr" in methods:
        all_results += extract_fast_hpo_cr(raw_text, hpo_labels, skip_negated)

    if "chatgpt" in methods:
        all_results += extract_chatgpt(raw_text, hpo_labels, skip_negated)

    if "phenobrain_api" in methods:
        all_results += extract_phenobrain_api(raw_text, hpo_labels, skip_negated)

    return deduplicate(all_results)


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_labels: Dict[str, str],
    methods: List[str] | None = None,
) -> Tuple[dict, List[dict]]:
    """
    Build a patient profile dict from raw clinical text.

    Runs extraction, then propagates extracted terms up the HPO ancestor
    hierarchy to include all parent terms (required for semantic similarity).

    Args:
        patient_id:  Unique patient identifier.
        raw_text:    Raw clinical text.
        hpo_labels:  Dict mapping HPO ID → label string.
        methods:     Extraction methods to use (see extract_hpo_terms).

    Returns:
        patient:         Dict with patient_id, raw_text, hpo_terms,
                         propagated_hpo_terms, methods_used.
        extracted_terms: List of dicts with full extraction provenance.
    """
    if methods is None:
        methods = ["dictionary"]

    extracted = extract_hpo_terms(
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
    )

    hpo_terms = sorted({r.hpo_id for r in extracted})

    # Propagate terms up the HPO ancestor hierarchy
    try:
        ancestors = load_json(HPO_ANCESTORS_PATH)
        ancestor_sets = preprocess_ancestor_sets(ancestors)
        propagated: set = set()
        for term in hpo_terms:
            propagated |= get_ancestors_inclusive(term, ancestor_sets)
        propagated_hpo_terms = sorted(propagated)
    except Exception as e:
        print(f"[ensemble] Warning: could not compute propagated terms: {e}")
        propagated_hpo_terms = hpo_terms

    patient = {
        "patient_id": patient_id,
        "raw_text": raw_text,
        "hpo_terms": hpo_terms,
        "propagated_hpo_terms": propagated_hpo_terms,
        "methods_used": list(methods),
    }

    return patient, [r.to_dict() for r in extracted]
