"""
Pipeline configuration and shared result-building utilities.
"""

from dataclasses import dataclass

from raresim.types.result import RunConfig, RunStats, SimilarityResult, MethodResults


@dataclass
class PipelineConfig:
    """
    Configuration for running similarity pipelines.
    Maps to the RunConfig schema for embedding in MethodResults.
    """

    top_k: int = 10
    use_propagated_terms: bool = True
    ic_threshold: float = 1.5
    use_canonical_profiles: bool = True

    @property
    def terms_key(self) -> str:
        """Helper to determine which HPO term set to use based on config."""
        return "propagated_hpo_terms" if self.use_propagated_terms else "hpo_terms"

    def to_run_config(self) -> RunConfig:
        """Convert to RunConfig for embedding in MethodResults."""
        return RunConfig(
            use_propagated_terms=self.use_propagated_terms,
            ic_threshold=self.ic_threshold,
            top_k=self.top_k,
            use_canonical_profiles=self.use_canonical_profiles,
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
