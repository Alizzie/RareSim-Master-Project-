"""
Shared utilities for running similarity pipelines, including a standardized main() function,
"""

from pathlib import Path

from shared.context import AppContext
from shared.io import load_patient, save_results, save_individual_results
from shared.paths import SHARED_DIR
from shared.result import Metadata, SimilarityResult
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for running similarity pipelines, with defaults that can be overridden."""

    top_k: int = 10
    use_propagated_terms: bool = True
    terms_key: str = "propagated_hpo_terms" if use_propagated_terms else "hpo_terms"
    ic_threshold: float = 1.5
    use_canonical_profiles: bool = True


def build_metadata(
    method_name: str,
    pipeline_name: str,
    config: PipelineConfig,
    n_patient_terms: int,
    n_disease_terms: int,
    ctx: AppContext,
) -> Metadata:
    """Helper to build standardized metadata for each similarity result."""
    return Metadata(
        method_name=method_name,
        pipeline_name=pipeline_name,
        use_propagated_terms=config.use_propagated_terms,
        ic_threshold=config.ic_threshold,
        top_k=config.top_k,
        n_patient_terms=n_patient_terms,
        n_disease_terms=n_disease_terms,
        app=ctx.app_metadata,
    )


def sort_and_rank(
    results: list[SimilarityResult],
    top_k: int,
) -> list[SimilarityResult]:
    """Sort by score descending, assign ranks, return top_k."""
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(sorted_results, start=1):
        result.rank = rank
    return sorted_results[:top_k]


def run_pipeline_main(
    pipeline_name: str,
    method_names: list[str],
    run_fn: callable,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> None:
    """
    Standardized main() for all pipelines.
    Handles loading, running, saving, and printing results.

    Usage in each pipeline:
        def main() -> None:
            run_pipeline_main(
                pipeline_name="tfidf",
                method_names=[METHOD_NAME],
                run_fn=run,
                output_dir=OUTPUT_DIR,
            )
    """
    if config is None:
        config = PipelineConfig()

    patient = load_patient(SHARED_DIR / "example_patient.json")
    ctx = AppContext.load(
        patient=patient, use_canonical_profiles=config.use_canonical_profiles
    )

    results = run_fn(patient, method_names, config, ctx)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir / f"{pipeline_name}_top{config.top_k}.json")

    for method_name, rows in results.items():
        save_individual_results(
            rows, output_dir / f"{method_name}_top{config.top_k}.json"
        )
        print(f"\nTop results for {method_name}:")
        for r in rows:
            print(
                f"  rank={r.rank:>2} | {r.disease_id:<15} | score={r.score:.4f} | {r.label}"
            )
