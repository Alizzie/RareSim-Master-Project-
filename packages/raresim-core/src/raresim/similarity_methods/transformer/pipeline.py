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

from raresim.utils.io import load_json, save_json
from raresim.utils.paths import (
    ALIAS_TO_CANONICAL_PATH,
    DISEASE_ANCESTORS_PATH,
    DISEASE_METADATA_INDEX_PATH,
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    PATIENT_PATH,
)
from raresim.utils.timer import timer
from raresim.similarity_methods.transformer.config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_MODEL_LIST,
    MODEL_LIST,
    TOP_K,
    TRANSFORMER_DIR,
)
from raresim.similarity_methods.transformer.methods import make_safe_model_name
from raresim.similarity_methods.transformer.retriever import DiseaseRetriever

PIPELINE_NAME = "transformer"


def run(  # pylint: disable=too-many-arguments,too-many-locals
    disease_profiles: dict,
    hpo_labels: dict,
    patient: dict,
    alias_to_canonical: dict,
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    model_list: list[str] | None = None,
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
        disease_ancestors:   ORDO ancestor chains for category paths.
        disease_metadata_index: ORDO labels/types for ancestor display.
        model_list:          List of model names to run.
        top_k:               Number of top results per model.
        candidate_pool_size: Candidates before canonical deduplication.
        rebuild_cache:       Force rebuild of embedding cache.

    Returns:
        Dict mapping model_name → list of ranked result dicts.
    """
    if model_list is None:
        model_list = list(MODEL_LIST)
    retriever = DiseaseRetriever(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=model_list,
        disease_ancestors=disease_ancestors,
        disease_metadata_index=disease_metadata_index,
        rebuild_cache=rebuild_cache,
    )


    print(f"\nPreparing cache for {len(model_list)} model(s)...")
    with timer("prepare transformer caches"):
        retriever.warmup(preload_models=False)


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



def run_default_model(  # pylint: disable=too-many-arguments
    disease_profiles: dict,
    hpo_labels: dict,
    patient: dict,
    alias_to_canonical: dict,
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    top_k: int = TOP_K,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    rebuild_cache: bool = False,
) -> dict[str, list[dict]]:
    """
    Run only the default transformer model.

    This is the preferred entry point for frontend/API requests because it
    avoids warming up every transformer model.
    """
    return run(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        patient=patient,
        alias_to_canonical=alias_to_canonical,
        disease_ancestors=disease_ancestors,
        disease_metadata_index=disease_metadata_index,
        model_list=DEFAULT_MODEL_LIST,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        rebuild_cache=rebuild_cache,
    )


def main() -> None:
    """Load shared artifacts and run the transformer retrieval pipeline."""
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
    disease_ancestors = load_json(DISEASE_ANCESTORS_PATH)
    disease_metadata_index = load_json(DISEASE_METADATA_INDEX_PATH)

    with timer("full transformer pipeline"):
        all_results = run(
            disease_profiles=disease_profiles,
            hpo_labels=hpo_labels,
            patient=patient,
            alias_to_canonical=alias_to_canonical,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
        )

    summary_path = TRANSFORMER_DIR / f"transformer_all_models_top{TOP_K}.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
