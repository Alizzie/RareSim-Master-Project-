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

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig, sort_and_rank
from raresim.similarity_methods.transformer.config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_MODEL_LIST,
    MODEL_LIST,
    PIPELINE_NAME,
    TRANSFORMER_DIR,
)
from raresim.similarity_methods.transformer.retriever import DiseaseRetriever
from raresim.types.result import MethodResults
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.timer import Timer, timer


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run transformer retrieval for selected models using direct HPO terms."""
    retriever = DiseaseRetriever.from_context(
        ctx=ctx,
        model_list=selected,
        patient=patient,
    )

    print(f"\nPreparing cache for {len(selected)} transformer model(s)...")
    with timer("prepare transformer caches"):
        retriever.warmup(preload_models=False)

    all_results: dict[str, MethodResults] = {}
    for model_name in selected:
        print(f"\nRunning model: {model_name}")
        model_timer = Timer(model_name).start()

        with timer(f"rank {model_name}"):
            rankings = retriever.rank(
                model_name=model_name,
                top_k=config.top_k,
                candidate_pool_size=CANDIDATE_POOL_SIZE,
            )

        elapsed = model_timer.stop()
        stats = retriever.run_stats(model_name, rankings, elapsed)

        all_results[model_name] = sort_and_rank(
            rankings,
            config,
            stats,
            model_name,
            PIPELINE_NAME,
        )

        retriever.unload_backend(model_name)
    return all_results


def run_default_model(
    patient: PatientProfile,
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """
    Run only the default transformer model.

    This is the preferred entry point for frontend/API requests because it
    avoids warming up every transformer model.
    """
    return run(patient, DEFAULT_MODEL_LIST, config, ctx)


def main() -> None:
    """Load shared artifacts and run the transformer retrieval pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=MODEL_LIST,
        run_fn=run,
        output_dir=TRANSFORMER_DIR,
    )


if __name__ == "__main__":
    main()
