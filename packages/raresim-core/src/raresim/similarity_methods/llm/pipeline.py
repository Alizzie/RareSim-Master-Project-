"""
LLM-based disease retrieval pipeline.

Directly asks biomedical LLMs to retrieve rare diseases from patient HPO terms
and explain why each disease matches the patient's phenotype profile.

Models (generative/decoder — not embedding models):
- Mistral/Mistral-7B-Instruct-v0.2
"""

from shared.io import load_json, save_json
from shared.paths import DISEASE_PROFILES_PATH, HPO_LABELS_PATH, PATIENT_PATH
from shared.timer import timer
from similarity_methods.llm.config import (
    LLM_DIR,
    LLM_MODEL_LIST,
    TOP_K,
)
from similarity_methods.llm.methods import retrieve_diseases_llm, unload_pipeline, explain_top_results

PIPELINE_NAME = "llm"


def run(
    patient: dict,
    hpo_labels: dict,
    disease_profiles: dict,
    model_list: list[str] = LLM_MODEL_LIST,
    top_k: int = TOP_K,
) -> dict[str, list[dict]]:
    """
    Run LLM disease retrieval for a patient across all models.

    Each model is loaded, run, then unloaded before the next starts
    to avoid GPU memory overflow on shared servers.

    Args:
        patient:          Patient dict with hpo_terms.
        hpo_labels:       HPO ID → label mapping.
        disease_profiles: Known disease profiles for validation.
        model_list:       List of HuggingFace model identifiers.
        top_k:            Number of diseases to return per model.

    Returns:
        Dict mapping model_name → list of ranked disease dicts.
    """
    all_results = {}

    for model_name in model_list:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        with timer(f"total {model_name}"):
            results, pipe = retrieve_diseases_llm(
                patient=patient,
                hpo_labels=hpo_labels,
                disease_profiles=disease_profiles,
                model_name=model_name,
                top_k=top_k,
            )

        # Unload immediately to free GPU memory for next model
        unload_pipeline(pipe)

        all_results[model_name] = results

        # Save per model
        safe_name = model_name.replace("/", "_")
        out_path = LLM_DIR / f"{safe_name}_top{top_k}.json"
        save_json(results, out_path)
        print(f"Saved to: {out_path}")

    return all_results


def main() -> None:
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)

    print(f"Patient  : {patient.get('patient_id')}")
    print(f"HPO terms: {patient.get('hpo_terms')}")
    print(f"Models   : {LLM_MODEL_LIST}")

    with timer("full LLM pipeline"):
        all_results = run(
            patient=patient,
            hpo_labels=hpo_labels,
            disease_profiles=disease_profiles,
        )

        # ── Add explanations to top results ──────────────────────────────
        print("\nRunning explainer on top results...")
        for model_name, results in all_results.items():
            if not results:
                continue
            print(f"\nExplaining results for: {model_name}")
            explained = explain_top_results(
                patient=patient,
                transformer_results=results,
                disease_profiles=disease_profiles,
                hpo_labels=hpo_labels,
            )
            all_results[model_name] = explained

            # Save per model after explanation
            safe_name = model_name.replace("/", "_")
            out_path = LLM_DIR / f"{safe_name}_top{TOP_K}.json"
            save_json(explained, out_path)
            print(f"Saved to: {out_path}")

        # Save combined summary
        summary_path = LLM_DIR / f"llm_all_models_top{TOP_K}.json"
        save_json(all_results, summary_path)
        print(f"\nSaved combined summary to: {summary_path}")

if __name__ == "__main__":
    main()
