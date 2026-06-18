"""
Template-based summary string generators.

Each function takes the already-built explanation components and produces
a single human-readable sentence describing the match. These are not
LLM-generated — they are deterministic string templates filled from
the structured data.

One generator per method family. All return a plain str.
"""

from raresim.core.explanation.schema import CoverageBlock


# ── Set-based ─────────────────────────────────────────────────────────────────
def set_based_summary(
    coverage: CoverageBlock,
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
    coverage: CoverageBlock,
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


# ── TF-IDF (placeholder — populated in the tfidf iteration) ─────────────────
def tfidf_summary(
    coverage: CoverageBlock,
    ic_weighted_score: float,
) -> str:
    """
    Example output:
        3 of 5 patient terms matched (60% patient / 43% disease coverage).
         TF-IDF weighted match score: 12.4."
    """
    pct_patient = round(coverage.patient_coverage * 100)
    pct_disease = round(coverage.disease_coverage * 100)
    return (
        f"{coverage.n_matched_terms} of {coverage.n_patient_terms} patient terms matched "
        f"({pct_patient}% patient / {pct_disease}% disease coverage). "
        f"TF-IDF weighted match score: {ic_weighted_score:.1f}."
    )


# ── Embedding methods (HPO2Vec / Autoencoder) — placeholders ─────────────────
def embedding_summary(
    coverage: CoverageBlock,
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
