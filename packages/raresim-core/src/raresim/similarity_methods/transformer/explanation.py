"""
Explanation utilities for transformer-based disease retrieval.

Transformer ranking is based on dense embedding cosine similarity.
The explanation describes the embedding-based score and shows shared HPO labels
only as supporting context.
"""


def get_shared_phenotype_labels(
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    hpo_labels: dict[str, str],
    top_n: int = 10,
) -> list[str]:
    """
    Return readable HPO labels shared by patient and disease.

    These labels are shown for interpretation only. The transformer score itself
    is computed from dense text embeddings, not direct HPO overlap.
    """
    shared_ids = set(patient_hpo_terms) & set(disease_hpo_terms)

    shared_labels = []
    for hpo_id in sorted(shared_ids):
        label = hpo_labels.get(hpo_id)
        if label:
            shared_labels.append(label.strip())

    return shared_labels[:top_n]


def build_explanation(
    *,
    score: float,
    model_name: str,
    model_type: str,
    patient_text: str,
    disease_text_preview: str,
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    hpo_labels: dict[str, str],
) -> dict:
    """
    Build a structured explanation for one transformer retrieval result.

    The score is embedding-based. Shared phenotype labels are included only to
    make the result easier to inspect clinically.
    """
    shared_labels = get_shared_phenotype_labels(
        patient_hpo_terms=patient_hpo_terms,
        disease_hpo_terms=disease_hpo_terms,
        hpo_labels=hpo_labels,
    )

    return {
        "summary": (
            f"Transformer embedding similarity ({model_name}): {score:.4f}. "
            "The score is based on dense text embeddings, not IC-based HPO scoring."
        ),
        "method": "transformer_cosine",
        "score": round(score, 6),
        "score_note": (
            "Cosine similarity between dense text embeddings. The patient profile "
            "and disease profile are encoded as vectors; higher scores indicate "
            "more similar text representations."
        ),
        "model_name": model_name,
        "model_type": model_type,
        "shared_phenotype_labels": shared_labels,
        "n_shared_phenotype_labels": len(shared_labels),
        "patient_text_preview": patient_text[:300],
        "disease_text_preview": disease_text_preview[:300],
        "interpretation_note": (
            "Shared phenotype labels are shown only for readability. They are not "
            "the direct scoring formula for the transformer method."
        ),
        "method_specific": {
            "embedding_method": "transformer_cosine",
            "embedding_input": "patient text and disease profile text",
            "uses_ic_values": False,
            "uses_direct_hpo_overlap_for_score": False,
        },
        "diagnostics": {
            "raw_score": round(score, 6),
        },
    }


def build_metadata(
    *,
    model_name: str,
    model_type: str,
    top_k: int,
) -> dict:
    """Build technical metadata for a transformer result."""
    return {
        "model_name": model_name,
        "model_type": model_type,
        "top_k": top_k,
        "embedding_normalization": "l2",
        "pooling": "mean" if model_type == "hf_encoder" else None,
        "sentence_transformer_encode": model_type == "sentence_transformer",
        "deduplicated_to_canonical": True,
        "method_family": "transformer",
    }
