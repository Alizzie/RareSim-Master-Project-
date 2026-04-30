"""
Transformer-based disease retrieval pipeline.

Embeds patient and disease texts using biomedical language models
and ranks diseases by cosine similarity of embeddings.

Models:
- PubMedBERT  : biomedical encoder, high scores but clustered
- ClinicalBERT: clinical notes encoder
- MiniLM      : general sentence transformer, more discriminative scores

Does not use run_pipeline_main() — transformer has a fundamentally different
run pattern (multiple models, embedding caches, canonical deduplication)
that the generic shared pipeline doesn't support.
"""

from shared.io import load_json, save_json
from shared.paths import (
    ALIAS_TO_CANONICAL_PATH,
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    PATIENT_PATH,
)
from similarity_methods.transformer.config import (
    CANDIDATE_POOL_SIZE,
    LLM_EXPLAINER_MODEL,
    MODEL_LIST,
    RUN_LLM_EXPLAINER,
    TOP_K,
    TOP_K_LLM_EXPLAIN,
    TRANSFORMER_DIR,
)
from similarity_methods.transformer.methods import make_safe_model_name
from similarity_methods.transformer.retriever import DiseaseRetriever
from similarity_methods.llm.methods import explain_top_results

PIPELINE_NAME = "transformer"


# ── LLM explanation ───────────────────────────────────────────────────────────


def run_llm_explanation(
    all_results: dict,
    patient: dict,
    disease_profiles: dict,
    hpo_labels: dict,
) -> None:
    """
    Add LLM explanations to transformer results in-place.
    Skipped if LLM explainer is unavailable.
    """
    try:
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

            for original, enriched in zip(results, explained):
                original["explanation"]["llm_reasoning"] = enriched.get(
                    "llm_explanation", ""
                )
                original["explanation"]["explainer_model"] = LLM_EXPLAINER_MODEL

    except ImportError as e:
        print(f"[transformer_pipeline] LLM explainer skipped: {e}")


# ── Main run function ─────────────────────────────────────────────────────────


def run(
    disease_profiles: dict,
    hpo_labels: dict,
    patient: dict,
    alias_to_canonical: dict,
    model_list: list[str] = MODEL_LIST,
    top_k: int = TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    rebuild_cache: bool = False,
) -> dict[str, list[dict]]:
    """
    Run transformer retrieval for all models.

    Args:
        disease_profiles:    Disease profiles dict.
        hpo_labels:          HPO ID → label mapping.
        patient:             Patient profile dict.
        alias_to_canonical:  Alias → canonical disease ID mapping.
        model_list:          List of model names to run.
        top_k:               Number of top results per model.
        candidate_pool_size: Candidates before canonical deduplication.
        rebuild_cache:       Force rebuild of embedding cache.

    Returns:
        Dict mapping model_name → list of ranked result dicts.
    """
    retriever = DiseaseRetriever(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=model_list,
        rebuild_cache=rebuild_cache,
    )

    retriever.warmup(preload_models=True)

    all_results = {}

    for model_name in model_list:
        print(f"\nRunning model: {model_name}")

        results = retriever.rank(
            model_name=model_name,
            patient=patient,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
        )

        all_results[model_name] = results

        safe_name = make_safe_model_name(model_name)
        out_path = TRANSFORMER_DIR / f"{safe_name}_top{top_k}_canonical.json"
        save_json(results, out_path)

        print(f"\nTop results for {model_name}:")
        for r in results:
            print(
                f"  rank={r['rank']:>2} | "
                f"{r['canonical_disease_id']:<15} | "
                f"score={r['score']:.4f} | "
                f"{r['label']}"
            )
        print(f"Saved to: {out_path}")

    return all_results


def main() -> None:
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)

    all_results = run(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        patient=patient,
        alias_to_canonical=alias_to_canonical,
    )

    if RUN_LLM_EXPLAINER:
        run_llm_explanation(
            all_results=all_results,
            patient=patient,
            disease_profiles=disease_profiles,
            hpo_labels=hpo_labels,
        )

    summary_path = TRANSFORMER_DIR / f"transformer_all_models_top{TOP_K}.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
