"""
HPO2Vec+ explanation builder.

- Build the method_specific block for the IC-weighted Word2Vec embedding method.
- Delegate all shared spine fields to core.explanation.base_explainer.

Covered fields in the explanation:
    embedding_method    — walk/embedding strategy identifier.
    aggregation         — how term vectors are combined (IC-weighted average).
    score_note          — what the cosine score is computed over.
    interpretation_note — clarifies matched terms are descriptive, not the score.

Inherited from base_explainer:
    coverage, matched_terms, unmatched_terms
"""

from raresim.core.explanation import build_base_explanation, ExplanationBlock


def _build_hpo2vec_summary(
    method_name: str,
    score: float,
    n_matched: int,
    n_patient_terms: int,
) -> str:
    """Human-readable one-liner for an HPO2Vec result."""
    return (
        f"HPO2Vec embedding similarity ({method_name}): {score:.4f}. "
        f"{n_matched} of {n_patient_terms} patient terms overlap directly. "
        "Score is cosine similarity of IC-weighted Word2Vec term embeddings, "
        "not direct HPO overlap."
    )


def build_explanation(  # pylint: disable=too-many-arguments
    *,
    method_name: str,
    score: float,
    patient_terms: set[str],
    disease_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    patient_raw_terms: set[str] | None = None,
    n_terms_in_vocab: int | None = None,
) -> dict:
    """
    Build a spine-conforming ExplanationBlock dict for one HPO2Vec result.

    Args:
        method_name:       e.g. "hpo2vec_plus".
        score:             Cosine similarity between patient and disease vectors.
        patient_terms:     Patient terms used for embedding.
        disease_terms:     Disease terms used for embedding.
        hpo_labels:        HPO ID -> label.
        ic_values:         HPO ID -> IC value.
        patient_raw_terms: Raw (non-propagated) terms for direct/propagated
                           classification in matched_terms.
        n_terms_in_vocab:  How many patient terms were actually in the Word2Vec
                           vocabulary (diagnostics only — terms outside the
                           vocab are silently skipped during embedding).
    """
    n_matched = len(patient_terms & disease_terms)

    summary = _build_hpo2vec_summary(
        method_name=method_name,
        score=score,
        n_matched=n_matched,
        n_patient_terms=len(patient_terms),
    )

    method_specific = {
        "embedding_method": "hpo2vec_random_walk",
        "aggregation": "ic_weighted_average",
        "uses_ic_values": True,
        "uses_dense_embeddings": True,
        "uses_direct_hpo_overlap_for_score": False,
        "score_note": (
            "Cosine similarity between IC-weighted averaged Word2Vec embeddings "
            "of the patient and disease HPO term sets."
        ),
        "interpretation_note": (
            "Matched-term and coverage fields are descriptive HPO overlap shown "
            "for readability. They do NOT drive the embedding score."
        ),
    }

    diagnostics: dict = {"raw_score": round(score, 6)}
    if n_terms_in_vocab is not None:
        diagnostics["n_patient_terms_in_vocab"] = n_terms_in_vocab

    explanation: ExplanationBlock = build_base_explanation(
        patient_terms=patient_terms,
        disease_terms=disease_terms,
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        patient_raw_terms=patient_raw_terms,
        method_specific=method_specific,
        diagnostics=diagnostics,
    )
    return explanation.to_dict()
