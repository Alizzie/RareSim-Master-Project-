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
    RUN_LLM_EXPLAINER,
    LLM_EXPLAINER_MODEL,
    TOP_K_LLM_EXPLAIN,
)

from transformer_embeddings import get_model_type, make_safe_model_name
from transformer_retriever import DiseaseRetriever, load_json, save_json

"""
Main pipeline for transformer-based disease retrieval.

Make sure to run this after you have the necessary data files
coming from build_shared_artifacts.py and the transformer models downloaded.

"""


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

    # ── LLM Explanation (if we want reasoning with LLM) ────────────────────────────────────────────
    if RUN_LLM_EXPLAINER:
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent.parent
            llm_dir = project_root / "pipelines" / "llm"

            sys.path.append(str(project_root))
            sys.path.append(str(llm_dir))    # ← this makes llm_explainer importable directly

            from pipelines.llm.llm_explainer import explain_top_results  

            for model_name, results in all_results.items():
                print(f"\nExplaining results for: {model_name}")

                explained = explain_top_results(
                    patient=patient,
                    transformer_results=results,
                    disease_profiles=disease_profiles,
                    hpo_labels=hpo_labels,
                    model_name=LLM_EXPLAINER_MODEL,
                    top_k=TOP_K_LLM_EXPLAIN,
                )

                for original, enriched in zip(all_results[model_name], explained):
                    original["explanation"]["llm_reasoning"] = enriched.get(
                        "llm_explanation", ""
                    )
                    original["explanation"]["explainer_model"] = LLM_EXPLAINER_MODEL

        except ImportError as e:
            print(f"[transformer_pipeline] LLM explainer skipped: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    summary_path = TRANSFORMER_DIR / "all_model_results_summary_canonical.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
