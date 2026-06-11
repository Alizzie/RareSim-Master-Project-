"""
Transformer-based disease retrieval pipeline.

Embeds patient and disease texts using biomedical language models
and ranks diseases by cosine similarity of embeddings.

Models (encoder-only, produce embeddings):
- PubMedBERT  : biomedical encoder, trained on PubMed abstracts
- ClinicalBERT: trained on clinical notes
- MiniLM      : lightweight general sentence transformer
- SapBERT     : trained for biomedical entity normalization
- BioBERT     : trained on PubMed abstracts and PMC full-text articles
"""

from shared.io import load_json, save_json
from shared.paths import (
    ALIAS_TO_CANONICAL_PATH,
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    PATIENT_PATH,
)
from shared.timer import timer
from similarity_methods.transformer.config import (
    CANDIDATE_POOL_SIZE,
    MODEL_LIST,
    TOP_K,
    TRANSFORMER_DIR,
)
from similarity_methods.transformer.methods import make_safe_model_name
from similarity_methods.transformer.retriever import DiseaseRetriever

PIPELINE_NAME = "transformer"


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

    print(f"\nWarming up {len(model_list)} models...")
    with timer("warmup all models"):
        retriever.warmup(preload_models=True)

    all_results = {}

    for model_name in model_list:
        print(f"\nRunning model: {model_name}")

        with timer(f"rank {model_name}"):
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

    with timer("full transformer pipeline"):
        all_results = run(
            disease_profiles=disease_profiles,
            hpo_labels=hpo_labels,
            patient=patient,
            alias_to_canonical=alias_to_canonical,
        )

    summary_path = TRANSFORMER_DIR / f"transformer_all_models_top{TOP_K}.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
