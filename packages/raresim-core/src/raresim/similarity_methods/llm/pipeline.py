"""
LLM-based disease retrieval pipeline.

Directly asks biomedical LLMs to retrieve rare diseases from patient HPO terms
and explain why each disease matches the patient's phenotype profile.

Models (generative/decoder — not embedding models):
- Mistral/Mistral-7B-Instruct-v0.2
"""

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.similarity_methods.llm.config import (
    LLM_DIR,
    LLM_MODEL_LIST,
    PIPELINE_NAME,
)
from raresim.types.result import MethodResults
from raresim.similarity_methods.llm.methods import unload_pipeline
from raresim.similarity_methods.llm.retriever import LlmDiseaseRetriever
from raresim.utils.timer import timer
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.types.schemas import PatientProfile


def run(  # pylint: disable=too-many-arguments
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """
    Run direct LLM disease retrieval + explanation for the patient.

    Each model is loaded, run, explained, and unloaded before the next starts
    to avoid GPU memory overflow on shared servers.

    Returns:
        Dictionary mapping model name to fully-explained MethodResults.
    """
    retriever = LlmDiseaseRetriever.from_context(patient, ctx)
    all_results: dict[str, MethodResults] = {}

    for model_name in selected:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        pipe = None
        try:
            with timer(f"total {model_name}"):
                results, pipe = retriever.retrieve(
                    model_name=model_name,
                    top_k=config.top_k,
                )
        finally:
            if pipe is not None:
                unload_pipeline(pipe)

        if results.rankings:
            print(f"\n[llm] Explaining top results for: {model_name}")
            results.rankings = retriever.explain_results(
                candidate_results=results.rankings
            )

        all_results[model_name] = results
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
