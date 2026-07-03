"""
LLM-based disease retrieval pipeline.

Directly asks biomedical LLMs to retrieve rare diseases from patient HPO terms
and explain why each disease matches the patient's phenotype profile.

Models (generative/decoder — not embedding models):
- Mistral/Mistral-7B-Instruct-v0.2
"""

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig, sort_and_rank
from raresim.similarity_methods.llm.config import (
    LLM_DIR,
    LLM_MODEL_LIST,
    MAX_NEW_TOKENS_RETRIEVAL,
    PIPELINE_NAME,
)
from raresim.types.result import MethodResults
from raresim.similarity_methods.llm.methods import unload_pipeline, load_hf_pipeline
from raresim.similarity_methods.llm.retriever import LlmDiseaseRetriever
from raresim.utils.timer import timer, Timer
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.types.schemas import PatientProfile


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run LLM retrieval/explanation for selected models."""
    retriever = LlmDiseaseRetriever.from_context(patient, ctx)
    all_results: dict[str, MethodResults] = {}

    for model_name in selected:
        print(f"\n{'=' * 60}\n  Model: {model_name}\n{'=' * 60}")
        method_timer = Timer(model_name).start()

        pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_RETRIEVAL)
        try:
            with timer(f"total {model_name}"):
                rankings = retriever.retrieve_with_pipe(
                    pipe, model_name=model_name, top_k=config.top_k
                )

            if rankings:
                print(f"\n[llm] Explaining top results for: {model_name}")
                rankings = retriever.explain_results_with_pipe(
                    pipe, candidate_results=rankings
                )
        finally:
            unload_pipeline(pipe)
            pipe = None

        elapsed = method_timer.stop()
        stats = retriever.run_stats(rankings, elapsed)
        all_results[model_name] = sort_and_rank(
            rankings, config, stats, model_name, PIPELINE_NAME
        )

    return all_results


def main() -> None:
    """Load shared artifacts and run the LLM retrieval/explanation pipeline."""

    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=LLM_MODEL_LIST,
        run_fn=run,
        output_dir=LLM_DIR,
    )


if __name__ == "__main__":
    main()
