"""
Explanation utilities for LLM-based disease retrieval.

LLM scores/rankings are generated from prompt-based semantic comparison.
These explanations are descriptive.
"""


def get_shared_phenotype_labels(
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    hpo_labels: dict[str, str],
    top_n: int = 10,
) -> list[str]:
    """
    Return readable HPO labels shared by patient and disease.

    This is supporting evidence only. The LLM ranking itself is not computed
    directly from this overlap unless a specific LLM method explicitly does so.
    """
    shared_ids = set(patient_hpo_terms) & set(disease_hpo_terms)

    shared_labels = []
    for hpo_id in sorted(shared_ids):
        label = hpo_labels.get(hpo_id)
        if label:
            shared_labels.append(label.strip())

    return shared_labels[:top_n]


def build_explanation(  # pylint: disable=too-many-arguments
    *,
    score: float,
    model_name: str,
    patient_text: str,
    disease_text_preview: str,
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    hpo_labels: dict[str, str],
    llm_response: str | None = None,
    prompt_name: str | None = None,
) -> dict:
    """Build a standardized explanation dictionary for one LLM disease result."""
    shared_labels = get_shared_phenotype_labels(
        patient_hpo_terms=patient_hpo_terms,
        disease_hpo_terms=disease_hpo_terms,
        hpo_labels=hpo_labels,
    )

    return {
        "summary": (
            f"LLM semantic ranking ({model_name}) assigned score {score:.4f}. "
            "The score is based on prompt-based semantic comparison, not "
            "IC-based HPO scoring."
        ),
        "method": "llm_ranking",
        "score": score,
        "score_note": (
            "LLM-based ranking score or confidence. The model compares the "
            "patient profile with disease profile information using prompt-based "
            "semantic reasoning."
        ),
        "model_name": model_name,
        "prompt_name": prompt_name,
        "shared_phenotype_labels": shared_labels,
        "n_shared_phenotype_labels": len(shared_labels),
        "patient_text_preview": patient_text[:300],
        "disease_text_preview": disease_text_preview[:300],
        "llm_response_preview": (llm_response or "")[:500],
        "interpretation_note": (
            "Shared phenotype labels are shown for readability. The LLM result "
            "should be interpreted as semantic support, not as a deterministic "
            "HPO-overlap score."
        ),
        "method_specific": {
            "llm_method": "prompt_based_ranking",
            "uses_ic_values": False,
            "uses_dense_embeddings": False,
        },
        "diagnostics": {
            "raw_score": round(score, 6),
        },
    }


def build_metadata(
    *,
    model_name: str,
    top_k: int,
    prompt_name: str | None = None,
) -> dict:
    """Build technical metadata for an LLM result."""
    return {
        "model_name": model_name,
        "top_k": top_k,
        "prompt_name": prompt_name,
        "deduplicated_to_canonical": True,
        "method_family": "llm",
    }
