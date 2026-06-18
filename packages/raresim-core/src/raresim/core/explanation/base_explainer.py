"""
Shared builder functions for the common explanation spine.

Every similarity pipeline calls these to populate the standardized
CoverageBlock, matched_terms, and unmatched_patient_terms fields.
Method-specific logic lives in each method's own explanation.py.

These functions are intentionally pure (no side effects, no I/O) so
they are easy to test in isolation.
"""

from raresim.core.explanation.schema import (
    CoverageBlock,
    ExplanationBlock,
    TermEntry,
    TermMatch,
)


def build_coverage_block(
    patient_terms: set[str],
    disease_terms: set[str],
) -> CoverageBlock:
    """
    Compute standardized coverage metrics for one (patient, disease) pair.

    Args:
        patient_terms: HPO terms used for the patient (may be propagated).
        disease_terms: HPO terms used for the disease (may be propagated).

    Returns:
        Populated CoverageBlock.
    """
    shared = patient_terms & disease_terms
    n_shared = len(shared)
    n_patient = len(patient_terms)
    n_disease = len(disease_terms)

    patient_cov = n_shared / n_patient if n_patient else 0.0
    disease_cov = n_shared / n_disease if n_disease else 0.0

    return CoverageBlock(
        patient_coverage=patient_cov,
        disease_coverage=disease_cov,
        n_patient_terms=n_patient,
        n_disease_terms=n_disease,
        n_matched_terms=n_shared,
        n_unmatched_patient_terms=len(patient_terms - disease_terms),
        direction_asymmetry=abs(patient_cov - disease_cov),
    )


def build_matched_terms(
    patient_terms: set[str],
    disease_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    patient_raw_terms: set[str] | None = None,
    match_scores: dict[str, float] | None = None,
    top_n: int = 15,
) -> list[TermMatch]:
    """
    Build the enriched matched-term list for the shared spine.

    Args:
        patient_terms:     Active patient term set (possibly propagated).
        disease_terms:     Active disease term set (possibly propagated).
        hpo_labels:        HPO ID -> human-readable label.
        ic_values:         HPO ID -> information content value.
        patient_raw_terms: If provided, terms present here are "direct";
                           those only in propagated set are "propagated".
                           Pass None to mark all matches as "direct".
        match_scores:      Optional HPO ID -> pairwise score (used by
                           semantic methods; set-based methods omit this).
        top_n:             Maximum number of matched terms to return,
                           sorted by IC descending.

    Returns:
        List of TermMatch objects, sorted by IC descending.
    """
    shared = patient_terms & disease_terms
    scores = match_scores or {}

    matched = [
        TermMatch(
            id=t,
            label=hpo_labels.get(t, t),
            ic=ic_values.get(t, 0.0),
            match_type=(
                "direct"
                if patient_raw_terms is None or t in patient_raw_terms
                else "propagated"
            ),
            match_score=scores.get(t, 1.0),
        )
        for t in shared
    ]

    return sorted(matched, key=lambda x: x.ic, reverse=True)[:top_n]


def build_unmatched_terms(
    patient_terms: set[str],
    disease_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    top_n: int = 10,
) -> list[TermEntry]:
    """
    Build the list of patient terms that have no match in the disease profile.

    These are clinically important: they may represent atypical features or
    potential exclusion criteria for this disease.

    Returns:
        List of TermEntry objects sorted by IC descending (most specific
        unmatched features first).
    """
    unmatched = patient_terms - disease_terms

    entries = [
        TermEntry(
            id=t,
            label=hpo_labels.get(t, t),
            ic=ic_values.get(t, 0.0),
        )
        for t in unmatched
    ]

    return sorted(entries, key=lambda x: x.ic, reverse=True)[:top_n]


def build_ic_filter_block(
    removed_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    terms_before: int,
    terms_after: int,
) -> dict:
    """
    Describe the impact of IC-threshold filtering on patient terms.

    Useful for semantic methods where terms below a threshold are dropped
    before scoring. Surfacing what was removed helps users understand why
    their patient's generic phenotypes did not contribute to the score.

    Args:
        removed_terms: Terms that were filtered out.
        hpo_labels:    HPO ID -> label.
        ic_values:     HPO ID -> IC value.
        terms_before:  Patient term count before filtering.
        terms_after:   Patient term count after filtering.

    Returns:
        Dict ready to embed under method_specific["ic_filter_impact"].
    """
    removed_entries = sorted(
        [
            {
                "id": t,
                "label": hpo_labels.get(t, t),
                "ic": round(ic_values.get(t, 0.0), 4),
            }
            for t in removed_terms
        ],
        key=lambda x: x["ic"],
    )

    return {
        "terms_before_filter": terms_before,
        "terms_after_filter": terms_after,
        "n_removed": len(removed_terms),
        "removed_terms": removed_entries,
    }


# ── Assembly helper ───────────────────────────────────────────────────────────


def build_base_explanation(
    patient_terms: set[str],
    disease_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    summary: str,
    patient_raw_terms: set[str] | None = None,
    match_scores: dict[str, float] | None = None,
    method_specific: dict | None = None,
    diagnostics: dict | None = None,
) -> ExplanationBlock:
    """
    Convenience function that assembles a complete ExplanationBlock
    from the shared components.

    Each method's explanation.py typically calls this after building
    its own method_specific dict, passing it in here.

    Args:
        patient_terms:    Active (possibly propagated) patient terms.
        disease_terms:    Active (possibly propagated) disease terms.
        hpo_labels:       HPO ID → label.
        ic_values:        HPO ID → IC.
        summary:          Human-readable one-liner (caller-generated).
        patient_raw_terms: Raw terms for direct/propagated classification.
        match_scores:     Per-term pairwise scores (semantic methods).
        method_specific:  Method-owned extension dict.
        diagnostics:      Debug-only data.

    Returns:
        Fully populated ExplanationBlock.
    """
    coverage = build_coverage_block(patient_terms, disease_terms)
    matched = build_matched_terms(
        patient_terms,
        disease_terms,
        hpo_labels,
        ic_values,
        patient_raw_terms=patient_raw_terms,
        match_scores=match_scores,
    )
    unmatched = build_unmatched_terms(
        patient_terms, disease_terms, hpo_labels, ic_values
    )

    return ExplanationBlock(
        summary=summary,
        coverage=coverage,
        matched_terms=matched,
        unmatched_patient_terms=unmatched,
        method_specific=method_specific or {},
        diagnostics=diagnostics or {},
    )
