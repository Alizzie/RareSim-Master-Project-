"""
Semantic similarity (BMA) explanation builder.

- Build the method_specific block for Resnik / Lin / Jiang-Conrath BMA.
- Delegate all shared spine fields to core.explanation.base_explainer.

Covered fields in the explanation:
    bma_directions      — p→d avg, d→p avg, asymmetry + interpretation.
    semantic_clusters   — group matched terms by their MICA.
    weak_patient_matches — patient terms that found poor BMA partners.
    ic_filter_impact    — terms removed by IC threshold filtering.

Inherited from base_explainer:
    coverage, matched_terms, unmatched_terms
"""

from collections import defaultdict

from raresim.core.explanation import (
    build_base_explanation,
    build_coverage_block,
    build_ic_filter_block,
    semantic_summary,
    ExplanationBlock,
)
from raresim.similarity_methods.semantic.config import WEAK_MATCH_THRESHOLD


def _interpret_asymmetry(p2d: float, d2p: float, threshold: float = 0.15) -> str:
    """
    Classify the bidirectional gap into a clinically readable label.

    symmetric              — both directions agree within threshold.
    patient_better_covered — patient terms match disease well, but disease
                             has many more terms the patient doesn't have.
    disease_better_covered — disease terms match patient well, but patient
                             has terms the disease doesn't cover.
    """
    gap = p2d - d2p
    if abs(gap) < threshold:
        return "symmetric"
    return "patient_better_covered" if gap > 0 else "disease_better_covered"


def _build_bma_directions(p2d_avg: float, d2p_avg: float) -> dict:
    return {
        "patient_to_disease_avg": round(p2d_avg, 4),
        "disease_to_patient_avg": round(d2p_avg, 4),
        "asymmetry": round(abs(p2d_avg - d2p_avg), 4),
        "asymmetry_interpretation": _interpret_asymmetry(p2d_avg, d2p_avg),
    }


def build_semantic_clusters(
    match_details: list[dict],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    min_cluster_size: int = 2,
    top_n: int = 5,
) -> list[dict]:
    """
    Group patient -> disease BMA matches by their shared MICA.

    When multiple patient terms route through the same MICA, they form a
    semantic cluster. This gives a higher-level view: instead of listing
    5 individual cerebellar terms, you see "5 terms cluster around Ataxia."

    Args:
        match_details:    List of per-term match dicts from best_match_scores().
                          Each dict has: source_term, best_target_term,
                          mica_term, score.
        hpo_labels:       HPO ID -> label.
        ic_values:        HPO ID -> IC.
        min_cluster_size: Only include clusters with at least this many
                          patient terms.
        top_n:            Maximum clusters to return, sorted by avg score.

    Returns:
        List of cluster dicts, sorted by cluster_avg_score descending.
    """
    buckets: dict[str, dict] = defaultdict(
        lambda: {"patient_terms": [], "disease_terms": [], "scores": []}
    )

    for m in match_details:
        mica = m.get("mica_term")
        if not mica:
            continue
        bucket = buckets[mica]
        bucket["patient_terms"].append(m["source_term"])
        if m.get("best_target_term"):
            bucket["disease_terms"].append(m["best_target_term"])
        bucket["scores"].append(m["score"])

    clusters = []
    for mica_id, data in buckets.items():
        if len(data["scores"]) < min_cluster_size:
            continue
        avg_score = sum(data["scores"]) / len(data["scores"])
        clusters.append(
            {
                "mica_id": mica_id,
                "mica_label": hpo_labels.get(mica_id, mica_id),
                "mica_ic": round(ic_values.get(mica_id, 0.0), 4),
                "n_patient_terms": len(set(data["patient_terms"])),
                "patient_terms": sorted(set(data["patient_terms"])),
                "disease_terms": sorted(set(data["disease_terms"])),
                "cluster_avg_score": round(avg_score, 4),
            }
        )

    return sorted(clusters, key=lambda x: x["cluster_avg_score"], reverse=True)[:top_n]


def _build_weak_matches(
    match_details: list[dict],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    threshold: float = WEAK_MATCH_THRESHOLD,
) -> list[dict]:
    """
    Identify patient terms whose best BMA partner scored below the threshold.

    These are the terms that "hurt" the similarity score — the patient
    has a phenotype the disease doesn't explain well. Clinically, they
    may be atypical features or point toward a different diagnosis.
    """
    weak = [
        {
            "id": m["source_term"],
            "label": hpo_labels.get(m["source_term"], m["source_term"]),
            "ic": round(ic_values.get(m["source_term"], 0.0), 4),
            "best_score": round(m["score"], 4),
            "best_match_id": m.get("best_target_term"),
            "best_match_label": hpo_labels.get(m.get("best_target_term", ""), ""),
            "mica_id": m.get("mica_term"),
            "mica_label": hpo_labels.get(m.get("mica_term", ""), ""),
        }
        for m in match_details
        if m["score"] < threshold
    ]
    # Sort by IC descending: high-IC weak matches are the most clinically notable
    return sorted(weak, key=lambda x: x["ic"], reverse=True)


def _build_match_score_index(match_details: list[dict]) -> dict[str, float]:
    """
    Build a {source_term: best_score} dict from p2d match details.
    Used to populate match_score on TermMatch objects in the shared spine.
    """
    return {m["source_term"]: m["score"] for m in match_details}


# ── Main builder ──────────────────────────────────────────────────────────────


def build_explanation(
    method_name: str,
    patient_terms: set[str],
    disease_terms: set[str],
    score: float,
    p2d_avg: float,
    d2p_avg: float,
    p2d_matches: list[dict],
    d2p_matches: list[dict],
    all_patient_terms_before_filter: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    ic_threshold: float | None,
    patient_raw_terms: set[str] | None = None,
) -> ExplanationBlock:
    """
    Build the complete ExplanationBlock for one semantic BMA result.

    Args:
        method_name:                    e.g. "semantic_resnik_bma".
        patient_terms:                  Patient terms after IC filtering.
        disease_terms:                  Disease terms after IC filtering.
        score:                          Final BMA score (average of both directions).
        p2d_avg:                        Patient-to-disease direction average.
        d2p_avg:                        Disease-to-patient direction average.
        p2d_matches:                    Per-term match details, patient→disease.
        d2p_matches:                    Per-term match details, disease→patient.
        all_patient_terms_before_filter: Patient terms before IC filtering,
                                         used to compute filter impact.
        hpo_labels:                     HPO ID -> label.
        ic_values:                      HPO ID -> IC.
        ic_threshold:                   The threshold that was applied.
        patient_raw_terms:              Raw (non-propagated) terms for
                                        direct/propagated classification.

    Returns:
        Fully populated ExplanationBlock.
    """
    # BMA direction block
    bma_directions = _build_bma_directions(p2d_avg, d2p_avg)

    # Semantic clusters from p->d direction
    clusters = build_semantic_clusters(p2d_matches, hpo_labels, ic_values)
    top_cluster_label = clusters[0]["mica_label"] if clusters else None

    # Weak matches from p->d direction
    weak_matches = _build_weak_matches(p2d_matches, hpo_labels, ic_values)

    # IC filter impact
    removed_by_filter = all_patient_terms_before_filter - patient_terms
    coverage_before = build_coverage_block(patient_terms, disease_terms)
    ic_filter_block = build_ic_filter_block(
        removed_terms=removed_by_filter,
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        terms_before=len(all_patient_terms_before_filter),
        terms_after=len(patient_terms),
    )

    method_specific = {
        "bma_variant": method_name.replace("semantic_", "").replace("_bma", ""),
        "bma_directions": bma_directions,
        "semantic_clusters": clusters,
        "weak_patient_matches": weak_matches,
        "ic_filter_impact": ic_filter_block,
    }

    # Summary string
    summary = semantic_summary(
        coverage=coverage_before,
        p2d_avg=p2d_avg,
        d2p_avg=d2p_avg,
        top_cluster_label=top_cluster_label,
        n_weak_matches=len(weak_matches),
        method_name=method_name,
    )

    # Per-term match scores for TermMatch objects (p->d direction)
    match_scores = _build_match_score_index(p2d_matches)

    return build_base_explanation(
        patient_terms=patient_terms,
        disease_terms=disease_terms,
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        patient_raw_terms=patient_raw_terms,
        match_scores=match_scores,
        method_specific=method_specific,
        diagnostics={
            "raw_score": round(score, 6),
            "ic_threshold_applied": ic_threshold,
            "n_p2d_matches": len(p2d_matches),
            "n_d2p_matches": len(d2p_matches),
        },
    )
