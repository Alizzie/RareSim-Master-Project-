"""
LLM-based disease retrieval pipeline.

Currently runs Mistral via Ollama (ollama backend). After getting gpu access, we can test more powerful Hugging Face models (hf backend).
to directly retrieve rare diseases from patient HPO terms.

Backend is controlled in config.py:
    LLM_BACKEND = "ollama"  ← local development (recommended)
    LLM_BACKEND = "hf"      ← GPU server with BioMistral or similar

Note: LLM explanation of transformer results is handled inside
transformer/pipeline.py when RUN_LLM_EXPLAINER = True.

Does not use run_pipeline_main() — LLM retrieval returns raw dicts
with fields (confidence, validated_against_profiles, ordo_id) that
don't fit the SimilarityResult schema.
"""

from shared.io import load_json, save_json
from shared.paths import DISEASE_PROFILES_PATH, HPO_LABELS_PATH, PATIENT_PATH
from similarity_methods.llm.config import (
    LLM_BACKEND,
    LLM_DIR,
    RETRIEVAL_MODEL,
    TOP_K,
)
from similarity_methods.llm.methods import retrieve_diseases_llm

PIPELINE_NAME = "llm"


def run(
    patient: dict,
    hpo_labels: dict,
    disease_profiles: dict,
    model_name: str = RETRIEVAL_MODEL,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run LLM disease retrieval for a patient.

    Args:
        patient:          Patient dict with hpo_terms.
        hpo_labels:       HPO ID → label mapping.
        disease_profiles: Known disease profiles for validation.
        model_name:       Model name for the active backend.
        top_k:            Number of diseases to return.

    Returns:
        List of ranked disease dicts.
    """
    return retrieve_diseases_llm(
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
        model_name=model_name,
        top_k=top_k,
    )


def main() -> None:
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)

    print(f"Patient:   {patient.get('patient_id')}")
    print(f"HPO terms: {patient.get('hpo_terms')}")
    print(f"Backend:   {LLM_BACKEND}")
    print(f"Model:     {RETRIEVAL_MODEL}\n")

    print("=" * 60)
    print("LLM Disease Retrieval")
    print("=" * 60)

    results = run(
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
    )

    out_path = LLM_DIR / f"llm_retrieval_top{TOP_K}.json"
    save_json(results, out_path)

    print(f"\nResults (top {TOP_K}):\n")
    for r in results:
        validated = "✓" if r["validated_against_profiles"] else "?"
        print(
            f"  rank={r['rank']:>2} | {validated} | "
            f"{r['ordo_id']:<15} | "
            f"conf={r['confidence']:<8} | "
            f"{r['disease_name']}"
        )

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
    