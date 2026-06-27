"""
Denoising autoencoder explanation builder.

- Build the method_specific block for the latent-embedding similarity method.
- Delegate all shared spine fields to core.explanation.base_explainer.

Covered fields in the explanation:
    embedding_method    — latent denoising-autoencoder embedding identifier.
    aggregation         — how terms enter the model (binary vocab vector).
    score_note          — what the euclidean similarity is computed over.
    interpretation_note — clarifies matched terms are descriptive, not the score.

Inherited from base_explainer:
    coverage, matched_terms, unmatched_terms
"""

from raresim.core.explanation import build_base_explanation, ExplanationBlock


def build_method_specific_explanation_block(
    method_specific_extra: dict | None = None,
) -> dict:
    """Build the method-specific explanation block for an LLM result."""
    return {
        "embedding_method": "denoising_autoencoder_latent",
        "aggregation": "binary_vocab_vector",
        "uses_ic_values": False,
        "uses_dense_embeddings": True,
        "uses_direct_hpo_overlap_for_score": False,
        "score_note": (
            "Similarity derived from L2 distance between the patient and disease "
            "binary HPO vectors after encoding through the trained denoising "
            "autoencoder: similarity = 1 / (1 + distance)."
        ),
        "interpretation_note": (
            "Matched-term and coverage fields are descriptive HPO overlap shown "
            "for readability. They do NOT drive the latent score."
        ),
        **(method_specific_extra or {}),
    }


def build_explanation(  # pylint: disable=too-many-arguments
    *,
    method_name: str,
    score: float,
    patient_terms: set[str],
    disease_terms: set[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    patient_raw_terms: set[str] | None = None,
    method_specific_extra: dict | None = None,
) -> dict:
    """Build a spine-conforming explanation dict for one autoencoder result."""
    n_matched = len(patient_terms & disease_terms)
    summary = (
        f"Denoising autoencoder latent similarity ({method_name}): {score:.4f}. "
        f"{n_matched} of {len(patient_terms)} patient terms overlap directly. "
        "Score is a euclidean-distance similarity between encoded latent "
        "vectors, not direct HPO overlap."
    )

    explanation: ExplanationBlock = build_base_explanation(
        patient_terms=patient_terms,
        disease_terms=disease_terms,
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        patient_raw_terms=patient_raw_terms,
        method_specific=build_method_specific_explanation_block(
            method_specific_extra=method_specific_extra
        ),
        diagnostics={"raw_score": round(score, 6)},
    )
    return explanation.to_dict()
