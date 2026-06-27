"""
Pipeline configuration and shared result-building utilities.
"""

from raresim.types.result import (
    RunStats,
    SimilarityResult,
    MethodResults,
    PipelineConfig,
)


def build_run_stats(
    n_patient_terms_raw: int,
    n_patient_terms_propagated: int,
    n_patient_terms_used: int,
    n_diseases_scored: int,
    n_diseases_skipped: int,
    computation_time: float = 0.0,
) -> RunStats:
    """
    Build a RunStats object from pipeline execution observations.

    Args:
        n_patient_terms_raw        : terms before propagation.
        n_patient_terms_propagated : terms after true-path propagation.
        n_patient_terms_used       : terms actually used for scoring
                                     (post IC-filter).
        n_diseases_scored          : profiles that produced a result.
        n_diseases_skipped         : profiles skipped (empty term set).
        computation_time           : scoring loop wall-clock time.
    """
    return RunStats(
        n_patient_terms_raw=n_patient_terms_raw,
        n_patient_terms_propagated=n_patient_terms_propagated,
        n_patient_terms_used=n_patient_terms_used,
        n_diseases_scored=n_diseases_scored,
        n_diseases_skipped=n_diseases_skipped,
        computation_time_seconds=computation_time,
    )


def sort_and_rank(
    results: list[SimilarityResult],
    config: PipelineConfig,
    stats: RunStats,
    method_name: str,
    pipeline_name: str,
) -> MethodResults:
    """
    Sort results by score descending, assign ranks, return top_k as MethodResults.

    Args:
        results       : unsorted list of SimilarityResult.
        config        : PipelineConfig for this run.
        stats         : RunStats observed during this run.
        method_name   : e.g. "set_jaccard".
        pipeline_name : e.g. "set_based".
    """
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(sorted_results, start=1):
        result.rank = rank
    return MethodResults(
        method_name=method_name,
        pipeline_name=pipeline_name,
        config=config.to_run_config(),
        stats=stats,
        rankings=sorted_results[: config.top_k],
    )
