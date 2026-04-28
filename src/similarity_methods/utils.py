"""Utility functions for similarity methods, including sorting and ranking results."""

from shared.result import SimilarityResult


def sort_and_rank(
    results: list[SimilarityResult], top_k: int
) -> list[SimilarityResult]:
    """Sort by score descending, assign ranks, return top_k."""
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    for rank, result in enumerate(sorted_results, start=1):
        result.rank = rank
    return sorted_results[:top_k]
