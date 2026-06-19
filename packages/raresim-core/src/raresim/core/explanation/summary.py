"""
Template-based summary string generators.

Each function takes the already-built explanation components and produces
a single human-readable sentence describing the match. These are not
LLM-generated — they are deterministic string templates filled from
the structured data.

One generator per method family. All return a plain str.
"""

from raresim.core.explanation.schema import (
    HpoCoverageBlock,
    TokenCoverageBlock,
    CoverageBlock,
)


# ── Set-based ─────────────────────────────────────────────────────────────────
def set_based_summary(
    coverage: HpoCoverageBlock,
    method_name: str,
    ic_weighted_score: float,
) -> str:
    """
    Example output:
        "3 of 5 patient terms matched (60% patient / 43% disease coverage).
         Match quality (IC-weighted): 12.4. Method: jaccard."
    """
    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)

    return (
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient terms matched "
        f"({pct_patient}% patient / {pct_disease}% disease coverage). "
        f"Match quality (IC-weighted): {ic_weighted_score:.1f}. "
        f"Method: {method_name}."
    )


# ── Semantic ──────────────────────────────────────────────────────────────────
def semantic_summary(
    coverage: HpoCoverageBlock,
    p2d_avg: float,
    d2p_avg: float,
    top_cluster_label: str | None,
    n_weak_matches: int,
    method_name: str,
) -> str:
    """
    Example output:
        "
        3 of 5 patient terms matched (60% patient / 43% disease coverage).
        Bidirectional BMA (resnik): p->d 0.72, d->p 0.64.
        Key shared concept: Cerebellar ataxia.
        1 patient term is atypical for this disease."
    """
    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)
    short_name = method_name.replace("semantic_", "").replace("_bma", "")
    parts = [
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient terms matched "
        f"({pct_patient}% patient / {pct_disease}% disease coverage). "
        f"Bidirectional BMA ({short_name}): "
        f"p->d {p2d_avg:.3f}, d->p {d2p_avg:.3f}."
    ]

    if top_cluster_label:
        parts.append(f"Key shared concept: {top_cluster_label}.")

    if n_weak_matches == 1:
        parts.append("1 patient term is atypical for this disease.")
    elif n_weak_matches > 1:
        parts.append(f"{n_weak_matches} patient terms are atypical for this disease.")

    return " ".join(parts)


# ── TF-IDF ─────────────────


def tfidf_summary(
    coverage: CoverageBlock,
    ic_weighted_score: float,
    mode: str = "tfidf_hpo",
    is_sparse: bool = False,
) -> str:
    """
    Produces a mode-specific summary string.

    Uses isinstance() to narrow the coverage type before accessing
    subtype-specific fields. This is safer than local re-annotation
    and lets the type checker verify correctness without type: ignore.
    """
    sparse_warning = (
        " Warning: disease description is very short — score may be unreliable."
        if is_sparse
        else ""
    )

    if mode == "tfidf_hpo":
        if not isinstance(coverage, HpoCoverageBlock):
            raise TypeError(
                f"tfidf_hpo mode requires HpoCoverageBlock, got {type(coverage).__name__}"
            )
        pct_patient = round(coverage.patient_coverage * 100)
        pct_disease = round(coverage.disease_coverage * 100)
        return (
            f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient HPO terms "
            f"matched ({pct_patient}% patient / {pct_disease}% disease coverage). "
            f"TF-IDF weighted match score: {ic_weighted_score:.1f}."
        )

    if not isinstance(coverage, TokenCoverageBlock):
        raise TypeError(
            f"{mode} requires TokenCoverageBlock, got {type(coverage).__name__}"
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
            f"{sparse_warning}"
        )

    # hybrid
    return (
        f"{n_matched} of {n_patient} patient HPO label tokens matched disease description "
        f"({pct_patient}% patient / {pct_disease}% disease token coverage, "
        f"{n_disease} disease tokens total). "
        f"See contributing_hpo_terms for phenotype-level traceability."
        f"{sparse_warning}"
    )


# ── Embedding methods (HPO2Vec / Autoencoder) — placeholders ─────────────────
def embedding_summary(
    coverage: HpoCoverageBlock,
    method_name: str,
    cosine_score: float,
) -> str:
    """
    Example output:
        "Embedding similarity (hpo2vec): 0.72. 3 of 5 patient terms
        directly overlap (60% patient / 43% disease coverage)."
    """
    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)
    return (
        f"Embedding similarity ({method_name}): {cosine_score:.4f}. "
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient terms "
        f"directly overlap ({pct_patient}% patient / {pct_disease}% disease coverage)."
    )
