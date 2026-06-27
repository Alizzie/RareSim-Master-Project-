"""
Set-based similarity explanation builder.

- Build the method_specific block for set-based methods
    (formula components + IC-weighted match quality).
- Delegate all shared spine fields to core.explanation.base_explainer.
- Produce the final ExplanationBlock via build_base_explanation.

Covered fields in the explanation:
    formula_components  — the raw numerator/denominator that produce the score.
    ic_weighted_score   — sum of IC values for matched terms (quality proxy).
    top_ic_matched      — top matched terms ranked by IC (most discriminating).

Inherited from base_explainer:
    coverage, matched_terms, unmatched_terms — those are in base_explainer.
"""

from raresim.core.explanation import (
    build_base_explanation,
    build_coverage_block,
    ExplanationBlock,
)

# ── Summary builder ────────────────────────────────────────────────


def _build_summary(
    patient_terms: set[str],
    disease_terms: set[str],
    method_name: str,
    ic_weighted_score: float,
) -> str:
    """
    Build the human-readable summary for a set-based result.

    Surfaces the three values that matter most for this method:
    matched term count, coverage in both directions, and IC-weighted
    match quality (a proxy for how specific the shared terms are).
    """
    coverage = build_coverage_block(patient_terms, disease_terms)
    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)
    short = method_name.replace("set_", "")
    return (
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient terms matched "
        f"({pct_patient}% patient / {pct_disease}% disease coverage). "
        f"IC-weighted match quality: {ic_weighted_score:.1f}. "
        f"Method: {short}."
    )


# ── Formula component builders ────────────────────────────────────────────────


def _jaccard_components(pat: set[str], disease: set[str]) -> dict:
    intersection = len(pat & disease)
    union = len(pat | disease)
    return {
        "formula": "jaccard",
        "intersection_size": intersection,
        "union_size": union,
    }


def _dice_components(pat: set[str], disease: set[str]) -> dict:
    intersection = len(pat & disease)
    return {
        "formula": "dice",
        "intersection_size": intersection,
        "size_patient": len(pat),
        "size_disease": len(disease),
    }


def _overlap_components(pat: set[str], disease: set[str]) -> dict:
    intersection = len(pat & disease)
    return {
        "formula": "overlap_coefficient",
        "intersection_size": intersection,
        "min_size": min(len(pat), len(disease)),
    }


def _cosine_components(pat: set[str], disease: set[str]) -> dict:
    # For binary cosine: dot = |intersection|, norms = sqrt(|A|) * sqrt(|B|)
    intersection = len(pat & disease)
    return {
        "formula": "cosine",
        "intersection_size": intersection,
        "size_patient": len(pat),
        "size_disease": len(disease),
    }


_COMPONENT_BUILDERS = {
    "set_jaccard": _jaccard_components,
    "set_dice": _dice_components,
    "set_overlap": _overlap_components,
    "set_cosine": _cosine_components,
}


# ── IC quality metrics ────────────────────────────────────────────────────────


def _ic_quality_block(
    patient_terms: set[str],
    disease_terms: set[str],
    ic_values: dict[str, float],
    top_n: int = 5,
) -> tuple[float, list[dict]]:
    """
    Compute IC-based match quality metrics for the shared terms.

    Returns:
        ic_weighted_score : sum of IC for all matched terms.
        top_ic_matched    : top_n matched terms sorted by IC descending.
    """
    shared = patient_terms & disease_terms
    ic_weighted_score = sum(ic_values.get(t, 0.0) for t in shared)
    top_ic = sorted(
        [{"id": t, "ic": round(ic_values.get(t, 0.0), 4)} for t in shared],
        key=lambda x: x["ic"],
        reverse=True,
    )[:top_n]
    return round(ic_weighted_score, 4), top_ic


# ── Main builder ──────────────────────────────────────────────────────────────


def build_explanation(
    method_name: str,
    patient_terms: set[str],
    disease_terms: set[str],
    score: float,
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    patient_raw_terms: set[str] | None = None,
) -> ExplanationBlock:
    """
    Build the complete ExplanationBlock for one set-based result.

    Args:
        method_name:       One of set_jaccard, set_dice, set_overlap, set_cosine.
        patient_terms:     Active (possibly propagated) patient terms.
        disease_terms:     Active (possibly propagated) disease terms.
        score:             The similarity score already computed by the pipeline.
        hpo_labels:        HPO ID → human-readable label.
        ic_values:         HPO ID → IC value.
        patient_raw_terms: Raw (non-propagated) patient terms for
                           direct vs propagated match classification.
                           Pass None to skip this classification.

    Returns:
        Fully populated ExplanationBlock.
    """
    # Method-specific: formula components
    component_fn = _COMPONENT_BUILDERS.get(method_name, _jaccard_components)
    formula_components = component_fn(patient_terms, disease_terms)

    # Method-specific: IC quality
    ic_weighted_score, top_ic_matched = _ic_quality_block(
        patient_terms, disease_terms, ic_values
    )

    method_specific = {
        "formula_components": formula_components,
        "ic_weighted_match_score": ic_weighted_score,
        "top_ic_matched_terms": top_ic_matched,
    }

    summary = _build_summary(
        patient_terms=patient_terms,
        disease_terms=disease_terms,
        method_name=method_name.replace("set_", ""),
        ic_weighted_score=ic_weighted_score,
    )

    return build_base_explanation(
        patient_terms=patient_terms,
        disease_terms=disease_terms,
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        patient_raw_terms=patient_raw_terms,
        match_scores=None,  # set-based: binary, no per-term score
        method_specific=method_specific,
        diagnostics={"raw_score": round(score, 6)},
    )
