from transformer_config import (
    PROJECT_ROOT,
    OUTPUTS_DIR,
    TRANSFORMER_DIR,
    SHARED_DIR,
    ALIAS_TO_CANONICAL_PATH,
    CANDIDATE_POOL_SIZE,
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    MODEL_LIST,
    PATIENT_PATH,
    TOP_K,
)

from transformer_embeddings import get_model_type, make_safe_model_name
from transformer_retriever import DiseaseRetriever, load_json, save_json
'''Main pipeline for transformer-based disease retrieval. make sure to run this after you have the necessary data files coming from build_shared_artifacts.py and the transformer models downloaded.'''

def main():
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)

    retriever = DiseaseRetriever(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=MODEL_LIST,
        rebuild_cache=False,
    )

    retriever.warmup(preload_models=True)

    all_results = {}

    for model_name in MODEL_LIST:
        print(f"\nRunning model: {model_name}")
        print(f"Model type: {get_model_type(model_name)}")

        results = retriever.rank(
            model_name=model_name,
            patient=patient,
            top_k=TOP_K,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
        )

        safe_name = make_safe_model_name(model_name)
        out_path = TRANSFORMER_DIR / f"{safe_name}_top{TOP_K}_canonical.json"
        save_json(results, out_path)

        all_results[model_name] = results

        for r in results:
            print(
                f"rank={r['rank']:>2} | "
                f"{r['canonical_disease_id']:<15} | "
                f"score={r['score']:.4f} | "
                f"{r['label']} | "
                f"aliases={len(r['matched_aliases'])}"
            )

        print(f"Saved to: {out_path}")

    summary_path = TRANSFORMER_DIR / "all_model_results_summary_canonical.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
    