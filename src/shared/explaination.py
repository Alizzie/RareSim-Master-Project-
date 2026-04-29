"""
Explanation expanders for similarity results.
Each expander adds a specific piece of information to an existing explanation dict.
Compose them by passing the result of one into the next, or use expand() to
apply multiple expanders at once.

Usage:
    # single expander
    explanation = with_shared_terms(explanation, patient_terms, disease_terms)

    # compose multiple
    explanation = expand(
        explanation,
        patient_terms,
        disease_terms,
        expanders=[with_shared_terms, with_coverage, with_term_counts],
    )
"""

from typing import Callable

ExplanationExpander = Callable[[dict, set[str], set[str]], dict]


# ── Individual expanders ──────────────────────────────────────────────────────


def with_shared_terms(
    explanation: dict,
    patient_terms: set[str],
    disease_terms: set[str],
    top_n: int = 10,
) -> dict:
    """Add the top shared HPO terms between patient and disease."""
    shared = sorted(patient_terms & disease_terms)
    explanation["top_shared_terms"] = shared[:top_n]
    explanation["n_shared_terms"] = len(shared)
    return explanation


def with_coverage(
    explanation: dict,
    patient_terms: set[str],
    disease_terms: set[str],
) -> dict:
    """
    Add coverage metrics:
    - patient_coverage: % of patient terms found in disease
    - disease_coverage: % of disease terms found in patient
    """
    shared = patient_terms & disease_terms
    explanation["patient_coverage"] = (
        len(shared) / len(patient_terms) if patient_terms else 0.0
    )
    explanation["disease_coverage"] = (
        len(shared) / len(disease_terms) if disease_terms else 0.0
    )
    return explanation


def with_term_counts(
    explanation: dict,
    patient_terms: set[str],
    disease_terms: set[str],
) -> dict:
    """Add raw term count information."""
    explanation["n_patient_terms"] = len(patient_terms)
    explanation["n_disease_terms"] = len(disease_terms)
    explanation["n_union_terms"] = len(patient_terms | disease_terms)
    return explanation


def with_unmatched_terms(
    explanation: dict,
    patient_terms: set[str],
    disease_terms: set[str],
    top_n: int = 10,
) -> dict:
    """Add patient terms not found in the disease profile."""
    unmatched = sorted(patient_terms - disease_terms)
    explanation["unmatched_patient_terms"] = unmatched[:top_n]
    explanation["n_unmatched_patient_terms"] = len(unmatched)
    return explanation


# ── Composer ──────────────────────────────────────────────────────────────────


def expand(
    explanation: dict,
    patient_terms: set[str],
    disease_terms: set[str],
    expanders: list[ExplanationExpander],
) -> dict:
    """Apply a list of expanders to an explanation dict in order."""
    for expander in expanders:
        explanation = expander(explanation, patient_terms, disease_terms)
    return explanation


# ── Preset combinations ───────────────────────────────────────────────────────

SET_BASED_EXPLANATION = [with_shared_terms, with_coverage, with_term_counts]
FULL_EXPLANATION = [
    with_shared_terms,
    with_coverage,
    with_term_counts,
    with_unmatched_terms,
]
MINIMAL_EXPLANATION = [with_shared_terms]
