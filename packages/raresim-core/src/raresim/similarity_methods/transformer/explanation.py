"""
Explanation utilities for transformer-based disease retrieval.

Transformer ranking is based on dense embedding cosine similarity.
The explanation describes the embedding-based score and shows shared HPO labels
only as supporting context.
"""

from typing import Any
from raresim.core.explanation import build_base_explanation


def build_method_specific_explanation_block(  # pylint: disable=too-many-arguments
    *,
    method_name: str,
    model_name: str,
    model_type: str,
    patient_text: str,
    disease_text_preview: str,
    method_specific_extra: dict | None = None,
) -> dict[str, Any]:
    """Build the method-specific block for a transformer result."""
    return {
        "embedding_method": method_name,
        "embedding_input": "patient text and disease profile text",
        "embedding_normalization": "l2",
        "pooling": "mean" if model_type == "hf_encoder" else None,
        "sentence_transformer_encode": model_type == "sentence_transformer",
        "model_name": model_name,
        "model_type": model_type,
        "uses_ic_values": False,
        "uses_dense_embeddings": True,
        "uses_direct_hpo_overlap_for_score": False,
        "deduplicated_to_canonical": True,
        "score_note": (
            "Cosine similarity between dense text embeddings. Patient and disease "
            "profiles are encoded as vectors; higher scores indicate more similar "
            "text representations."
        ),
        "interpretation_note": (
            "Coverage and matched-term fields are descriptive HPO overlap on raw "
            "terms, shown for readability. They do NOT drive the embedding score "
            "and are not comparable to the propagated-term coverage of semantic "
            "methods."
        ),
        "patient_text_preview": patient_text[:300],
        "disease_text_preview": disease_text_preview[:300],
        **(method_specific_extra or {}),
    }


def build_explanation(  # pylint: disable=too-many-arguments
    *,
    score: float,
    model_name: str,
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    ic_values: dict[str, float],
    hpo_labels: dict[str, str],
    method_specific: dict[str, Any] | None = None,
    diagnostics_extras: dict[str, Any] | None = None,
) -> dict:
    """
    Build a structured explanation for one transformer retrieval result.

    The score is embedding-based. Shared phenotype labels are included only to
    make the result easier to inspect clinically.
    """

    summary = (
        f"Transformer embedding similarity ({model_name}): {score:.4f}. "
        "The score is based on dense text embeddings, not IC-based HPO scoring."
    )

    return build_base_explanation(
        patient_terms=set(patient_hpo_terms),
        disease_terms=set(disease_hpo_terms),
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        method_specific=method_specific,
        diagnostics={"raw_score": round(score, 6), **(diagnostics_extras or {})},
    ).to_dict()
