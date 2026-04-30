"""
Reusable similarity methods with explanations.
All methods wrap the pure math from shared.math and add an explanation dict.
Accepts both sets and dicts (binary vectors) as input.
"""

from typing import Dict, List, Set, Tuple
from shared.math import (
    TermInput,
    PairwiseSimilarityFn,
    cosine_similarity as _cosine,
    to_binary_vector,
)


def _empty_explanation(method: str, pat: TermInput) -> dict:
    """Shared explanation for empty input cases."""
    return {
        "method": method,
        "failure_reason": f"{'Patient' if not pat else 'Disease'} input is empty.",
    }


def _shared_terms(pat: TermInput, disease: TermInput) -> list[str]:
    a = set(to_binary_vector(pat).keys())
    b = set(to_binary_vector(disease).keys())
    shared_terms = a & b
    return sorted(shared_terms)[:10], len(shared_terms)


def cosine_similarity(
    pat: TermInput, disease: TermInput, use_binary: bool = False
) -> Tuple[float, dict]:
    """
    Cosine similarity between two vectors with an explanation of the calculation steps.
    """
    if not pat or not disease:
        return 0.0, _empty_explanation("cosine_similarity", pat)

    score = _cosine(pat, disease, use_binary)
    shared_terms, n_shared = _shared_terms(pat, disease)
    explanation = {
        "method": "cosine_similarity",
        "score": score,
        "top_shared_terms": shared_terms,
        "n_shared_terms": n_shared,
    }
    return score, explanation


def best_match_scores(
    source_terms: Set[str],
    target_terms: Set[str],
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
    similarity_fn: PairwiseSimilarityFn,
) -> Tuple[float, List[dict]]:
    """
    Compute Best Match Average (BMA) scores from source terms to target terms.

    For each source term, finds the target term with the highest pairwise
    similarity. Returns the average best-match score and per-term match details.

    Used by the semantic similarity pipeline for Resnik, Lin, Jiang-Conrath BMA.

    Args:
        source_terms:   Set of HPO terms to match from.
        target_terms:   Set of HPO terms to match against.
        ancestor_sets:  Preprocessed inclusive ancestor sets.
        ic_values:      Dict mapping HPO ID → IC value.
        similarity_fn:  Pairwise similarity function (e.g. resnik_similarity).

    Returns:
        Tuple of:
        - average_score: mean of best-match scores across all source terms
        - match_details: list of dicts with source, best target, MICA, score
    """
    if not source_terms or not target_terms:
        return 0.0, []

    match_details = []

    for source_term in source_terms:
        best_score = 0.0
        best_target = None
        best_mica = None

        for target_term in target_terms:
            score, mica = similarity_fn(
                source_term,
                target_term,
                ancestor_sets,
                ic_values,
            )
            if score > best_score:
                best_score = score
                best_target = target_term
                best_mica = mica

        match_details.append(
            {
                "source_term": source_term,
                "best_target_term": best_target,
                "mica_term": best_mica,
                "score": best_score,
            }
        )

    average_score = sum(x["score"] for x in match_details) / len(match_details)
    return average_score, match_details
