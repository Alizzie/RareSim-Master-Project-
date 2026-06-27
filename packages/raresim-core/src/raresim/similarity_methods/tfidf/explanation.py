"""
TF-IDF similarity explanation builder.

Covered fields in the explanation:
  idf_weighted_matches   — matched terms ranked by IDF weight
  low_idf_matches        — matched terms that contributed little
  vector_norm_info       — patient and disease vector norms
  score_decomposition    — dot product and norm breakdown

Inherited from base_explainer:
  coverage, matched_terms, unmatched_terms
"""

import math

from raresim.core.explanation import (
    build_base_explanation,
    build_base_token_explanation,
    build_coverage_block,
    build_token_coverage_block,
    ExplanationBlock,
)
from raresim.similarity_methods.tfidf.config import (
    LOW_IDF_THRESHOLD,
    METHOD_HPO,
    METHOD_HYBRID,
    METHOD_HPO_LABELS,
    SPARSE_DISEASE_THRESHOLD,
)
from raresim.utils.text_utils import tokenize
from raresim.core.explanation import build_ic_filter_block


def _build_hpo_summary(
    patient_terms: set[str],
    disease_terms: set[str],
    ic_weighted_score: float,
) -> str:
    """
    Produces a mode-specific summary string.

    Uses isinstance() to narrow the coverage type before accessing
    subtype-specific fields. This is safer than local re-annotation
    and lets the type checker verify correctness without type: ignore.
    """
    coverage = build_coverage_block(patient_terms, disease_terms)

    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)
    return (
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient HPO terms "
        f"matched ({pct_patient}% patient / {pct_disease}% disease coverage). "
        f"TF-IDF weighted match score: {ic_weighted_score:.1f}."
    )


def _build_token_summary(
    patient_vec: dict[str, float],
    disease_vec: dict[str, float],
    weighted_score: float,
    mode: str = METHOD_HYBRID,
) -> str:
    """
    Produces a mode-specific summary string.

    Uses isinstance() to narrow the coverage type before accessing
    subtype-specific fields. This is safer than local re-annotation
    and lets the type checker verify correctness without type: ignore.
    """

    coverage = build_token_coverage_block(
        patient_vec, disease_vec, SPARSE_DISEASE_THRESHOLD
    )

    is_sparse = coverage.is_sparse_disease
    sparse_warning = (
        " Warning: disease description is very short — score may be unreliable."
        if is_sparse
        else ""
    )

    pct_patient = round(coverage.patient_token_coverage * 100)
    pct_disease = round(coverage.disease_token_coverage * 100)
    n_matched = coverage.n_matched_tokens
    n_patient = coverage.n_patient_tokens
    n_disease = coverage.n_disease_tokens

    if mode == "tfidf_text":
        return (
            f"{n_matched} of {n_patient} patient text tokens matched disease description "
            f"({pct_patient}% patient / {pct_disease}% disease token coverage, "
            f"{n_disease} disease tokens total)."
            f"Weighted TFIDF match score: {weighted_score:.1f}."
            f"{sparse_warning}"
        )

    if mode == "tfidf_label":
        return (
            f"{n_matched} of {n_patient} patient HPO label tokens matched disease HPO labels "
            f"({pct_patient}% patient / {pct_disease}% disease token coverage, "
            f"{n_disease} disease tokens total)."
            f"Weighted TFIDF match score: {weighted_score:.1f}."
            f"{sparse_warning}"
        )

    # hybrid
    return (
        f"{n_matched} of {n_patient} patient HPO label tokens matched disease description "
        f"({pct_patient}% patient / {pct_disease}% disease token coverage, "
        f"{n_disease} disease tokens total). "
        f"Weighted TFIDF match score: {weighted_score:.1f}."
        f"See contributing_hpo_terms for phenotype-level traceability."
        f"{sparse_warning}"
    )


# ── Vector norm helpers ───────────────────────────────────────────────────────


def _vector_norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(v**2 for v in vec.values()))


def _dot_product(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    return sum(vec_a[t] * vec_b[t] for t in vec_a if t in vec_b)


def _build_norm_block(
    patient_vec: dict[str, float],
    disease_vec: dict[str, float],
) -> dict:
    """
    Expose the vector norm components that produce the cosine score.

    A high patient norm means the patient has many specific terms (high IDF).
    A high disease norm means the disease has a broad, specific phenotype profile.
    The dot product is the raw overlap before normalisation.
    """
    dot = _dot_product(patient_vec, disease_vec)
    pat_norm = _vector_norm(patient_vec)
    dis_norm = _vector_norm(disease_vec)

    return {
        "dot_product": round(dot, 6),
        "patient_norm": round(pat_norm, 6),
        "disease_norm": round(dis_norm, 6),
        "score_check": round(
            dot / (pat_norm * dis_norm) if pat_norm > 0 and dis_norm > 0 else 0.0, 6
        ),
    }


# ── IDF match analysis ────────────────────────────────────────────────────────


def _build_low_idf_matches(
    patient_vec: dict[str, float],
    disease_vec: dict[str, float],
    idf: dict[str, float],
    hpo_labels: dict[str, str],
    low_idf_threshold: float = LOW_IDF_THRESHOLD,
    is_hpo_mode: bool = True,
) -> list[dict]:
    """
    Return matched terms whose IDF weight falls below the threshold.
    """
    shared_keys = set(patient_vec.keys()) & set(disease_vec.keys())
    low_idf = []
    for t in shared_keys:
        weight = round(idf.get(t, 0.0), 4)
        if weight < low_idf_threshold:
            if is_hpo_mode:
                low_idf.append(
                    {
                        "id": t,
                        "label": hpo_labels.get(t, t),
                        "idf_weight": weight,
                    }
                )
            else:
                low_idf.append({"token": t, "idf_weight": weight})
    return sorted(low_idf, key=lambda x: x["idf_weight"])


# ── HPO terms that contributed tokens (hybrid mode) ──────────────────────────


def _contributing_hpo_terms(
    patient_hpo_terms: set[str],
    disease_vec: dict[str, float],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    idf: dict[str, float],
) -> list[dict]:
    """
    For hybrid mode: identify which patient HPO terms contributed tokens
    that matched words in the disease description.
    """
    result = []
    for term in patient_hpo_terms:
        label = hpo_labels.get(term, term)

        label_tokens = set(tokenize(label))
        matching_tokens = [t for t in label_tokens if t in disease_vec]
        if matching_tokens:
            result.append(
                {
                    "hpo_id": term,
                    "hpo_label": label,
                    "ic": round(ic_values.get(term, 0.0), 4),
                    "matching_tokens": sorted(matching_tokens),
                    "token_idf_weights": {
                        t: round(idf.get(t, 0.0), 4) for t in matching_tokens
                    },
                }
            )

    return sorted(result, key=lambda x: x["ic"], reverse=True)


# ── Main builder ──────────────────────────────────────────────────────────────


def build_explanation(
    tfidf_mode: str,
    patient_terms: set[str],
    disease_terms: set[str],
    patient_vec: dict[str, float],
    disease_vec: dict[str, float],
    idf: dict[str, float],
    score: float,
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    patient_raw_terms: set[str] | None = None,
    all_patient_terms_before_filter: set[str] | None = None,
    low_idf_threshold: float = LOW_IDF_THRESHOLD,
    extra_diagnostics: dict | None = None,
) -> ExplanationBlock:
    """
    Build the complete ExplanationBlock for one TF-IDF result.

    Args:
        patient_terms:     Active (possibly propagated) patient HPO terms.
        disease_terms:     Active (possibly propagated) disease HPO terms.
        patient_vec:       TF-IDF vector for the patient {term: idf_weight}.
        disease_vec:       TF-IDF vector for the disease {term: idf_weight}.
        idf:               Global IDF dict {term: idf_value}.
        score:             Cosine similarity score already computed.
        hpo_labels:        HPO ID → human-readable label.
        ic_values:         HPO ID → IC value (for shared spine matched_terms).
        patient_raw_terms: Raw (non-propagated) patient terms for
                           direct vs propagated classification.
        all_patient_terms_before_filter: All patient terms before any filtering.
        low_idf_threshold:  IDF weight below which a term is considered low-signal.
        extra_diagnostics: Additional diagnostic information.
    """

    norm_block = _build_norm_block(patient_vec, disease_vec)
    diagnostics = {
        "raw_score": round(score, 6),
        "tfidf_mode": tfidf_mode,
        "low_idf_threshold_applied": low_idf_threshold,
        "n_patient_vector_terms": len(patient_vec),
        "n_disease_vector_terms": len(disease_vec),
        **(extra_diagnostics or {}),
    }

    # ── HPO mode ──────────────────────────────────────────────────────────────
    if tfidf_mode == METHOD_HPO:
        shared_hpo = patient_terms & disease_terms
        ic_weighted_score = round(sum(ic_values.get(t, 0.0) for t in shared_hpo), 4)
        match_scores = {t: round(idf.get(t, 0.0), 4) for t in shared_hpo}

        low_idf = _build_low_idf_matches(
            patient_vec,
            disease_vec,
            idf,
            hpo_labels,
            low_idf_threshold=low_idf_threshold,
            is_hpo_mode=True,
        )

        method_specific = {
            "tfidf_mode": tfidf_mode,
            "low_idf_matches": low_idf,
            "n_low_idf_matches": len(low_idf),
            "low_idf_threshold": low_idf_threshold,
            "ic_weighted_match_score": ic_weighted_score,
            "vector_norms": norm_block,
        }

        summary = _build_hpo_summary(
            patient_terms=patient_terms,
            disease_terms=disease_terms,
            ic_weighted_score=ic_weighted_score,
        )

        return build_base_explanation(
            patient_terms=patient_terms,
            disease_terms=disease_terms,
            hpo_labels=hpo_labels,
            ic_values=ic_values,
            summary=summary,
            patient_raw_terms=patient_raw_terms,
            match_scores=match_scores,
            method_specific=method_specific,
            diagnostics=diagnostics,
        )

    # ── Text and hybrid modes ─────────────────────────────────────────────────
    low_idf = _build_low_idf_matches(
        patient_vec,
        disease_vec,
        idf,
        {},
        low_idf_threshold=low_idf_threshold,
        is_hpo_mode=False,
    )

    idf_weighted_score = round(
        sum(patient_vec[t] for t in patient_vec if t in disease_vec), 4
    )

    method_specific = {
        "tfidf_mode": tfidf_mode,
        "idf_weighted_score": idf_weighted_score,
        "low_idf_matches": low_idf,
        "n_low_idf_matches": len(low_idf),
        "low_idf_threshold": low_idf_threshold,
        "vector_norms": norm_block,
    }

    # Hybrid: add traceability from tokens back to patient HPO terms
    if (
        tfidf_mode == METHOD_HYBRID or tfidf_mode == METHOD_HPO_LABELS
    ) and patient_terms:
        method_specific["contributing_hpo_terms"] = _contributing_hpo_terms(
            patient_terms, disease_vec, hpo_labels, ic_values, idf
        )

        if all_patient_terms_before_filter is not None:
            removed = all_patient_terms_before_filter - patient_terms
            method_specific["ic_filter_impact"] = build_ic_filter_block(
                removed_terms=removed,
                hpo_labels=hpo_labels,
                ic_values=ic_values,
                terms_before=len(all_patient_terms_before_filter),
                terms_after=len(patient_terms),
            )

    summary = _build_token_summary(
        patient_vec=patient_vec,
        disease_vec=disease_vec,
        mode=tfidf_mode,
        weighted_score=idf_weighted_score,
    )

    return build_base_token_explanation(
        patient_vec=patient_vec,
        disease_vec=disease_vec,
        idf=idf,
        summary=summary,
        method_specific=method_specific,
        diagnostics=diagnostics,
    )
