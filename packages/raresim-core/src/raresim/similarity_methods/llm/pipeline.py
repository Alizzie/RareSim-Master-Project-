"""
LLM-based disease retrieval pipeline.

Directly asks biomedical LLMs to retrieve rare diseases from patient HPO terms
and explain why each disease matches the patient's phenotype profile.

Models (generative/decoder — not embedding models):
- Mistral/Mistral-7B-Instruct-v0.2
"""

from raresim.similarity_methods.llm.config import (
    LLM_DIR,
    LLM_MODEL_LIST,
    TOP_K,
)
from raresim.similarity_methods.llm.methods import unload_pipeline
from raresim.similarity_methods.llm.retriever import LlmDiseaseRetriever
from raresim.utils.io import load_json, save_json
from raresim.utils.paths import (
    DISEASE_ANCESTORS_PATH,
    DISEASE_METADATA_INDEX_PATH,
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    PATIENT_PATH,
)
from raresim.utils.timer import timer

PIPELINE_NAME = "llm"


def run(  # pylint: disable=too-many-arguments
    patient: dict,
    hpo_labels: dict,
    disease_profiles: dict,
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    model_list: list[str] | None = None,
    top_k: int = TOP_K,
) -> dict[str, list[dict]]:
    """
    Run direct LLM disease retrieval for a patient across all models.

    Each model is loaded, run, then unloaded before the next starts
    to avoid GPU memory overflow on shared servers.
    """
    if model_list is None:
        model_list = list(LLM_MODEL_LIST)

    retriever = LlmDiseaseRetriever(
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
        disease_ancestors=disease_ancestors,
        disease_metadata_index=disease_metadata_index,
    )

    all_results = {}

    for model_name in model_list:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        pipe = None
        try:
            with timer(f"total {model_name}"):
                results, pipe = retriever.retrieve(
                    model_name=model_name,
                    top_k=top_k,
                )
        finally:
            if pipe is not None:
                unload_pipeline(pipe)

        all_results[model_name] = results

        safe_name = model_name.replace("/", "_")
        out_path = LLM_DIR / f"{safe_name}_top{top_k}.json"
        save_json(results, out_path)
        print(f"Saved to: {out_path}")

    return all_results


def main() -> None:
    """Load shared artifacts and run the LLM retrieval/explanation pipeline."""
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)
    disease_ancestors = load_json(DISEASE_ANCESTORS_PATH)
    disease_metadata_index = load_json(DISEASE_METADATA_INDEX_PATH)

    retriever = LlmDiseaseRetriever(
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
        disease_ancestors=disease_ancestors,
        disease_metadata_index=disease_metadata_index,
    )

    print(f"Patient  : {patient.get('patient_id')}")
    print(f"HPO terms: {patient.get('hpo_terms')}")
    print(f"Models   : {LLM_MODEL_LIST}")

    with timer("full LLM pipeline"):
        all_results = run(
            patient=patient,
            hpo_labels=hpo_labels,
            disease_profiles=disease_profiles,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
        )

        print("\nRunning explainer on top results...")
        for model_name, results in all_results.items():
            if not results:
                continue

            print(f"\nExplaining results for: {model_name}")
            explained = retriever.explain_results(candidate_results=results)
            all_results[model_name] = explained

            safe_name = model_name.replace("/", "_")
            out_path = LLM_DIR / f"{safe_name}_top{TOP_K}.json"
            save_json(explained, out_path)
            print(f"Saved to: {out_path}")

        summary_path = LLM_DIR / f"llm_all_models_top{TOP_K}.json"
        save_json(all_results, summary_path)
        print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
