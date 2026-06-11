"""
Pipeline configuration and shared result-building utilities.
"""

from dataclasses import dataclass

from raresim.types.result import Metadata, SimilarityResult, MethodResults


@dataclass
class PipelineConfig:
    """Configuration for running similarity pipelines, with defaults that can be overridden."""

    top_k: int = 10
    use_propagated_terms: bool = True
    ic_threshold: float = 1.5
    use_canonical_profiles: bool = True

    @property
    def terms_key(self) -> str:
        """Helper to determine which HPO term set to use based on config."""
        return "propagated_hpo_terms" if self.use_propagated_terms else "hpo_terms"


def build_metadata(
    method_name: str,
    pipeline_name: str,
    config: PipelineConfig,
    n_patient_terms: int,
    n_disease_terms: int,
    computation_time: float = 0.0,
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
        computation_time=computation_time,
    )


def sort_and_rank(
    results: list[SimilarityResult],
    metadata: Metadata,
    top_k: int,
) -> MethodResults:
    """Sort by score descending, assign ranks, return top_k."""
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(sorted_results, start=1):
        result.rank = rank
    return MethodResults(metadata=metadata, rankings=sorted_results[:top_k])
