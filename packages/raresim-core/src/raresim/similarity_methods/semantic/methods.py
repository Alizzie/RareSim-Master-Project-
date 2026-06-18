"""
Module for semantic similarity methods between HPO terms and term sets.

Core idea:
- Terms are compared using the ontology structure (ancestors)
- Information Content (IC) is used to measure specificity
- More specific shared ancestors → higher similarity

Methods:
- Resnik    : IC(MICA) — specificity of shared ancestor
- Lin       : normalized Resnik — accounts for term specificity
- Jiang-Conrath : distance-based, converted to similarity

Performance:
- get_mica() results are cached at module level in _mica_cache.
- Since ancestor_sets and ic_values are fixed for the entire batch run,
  the same (term_a, term_b) pair always produces the same result.
- This avoids recomputing ancestor intersections across 20k diseases.
- Call clear_mica_cache() between patients if memory becomes a concern.
"""

import math
from typing import Callable

from raresim.utils.hpo_utils import get_ancestors_inclusive

# ── Type alias ────────────────────────────────────────────────────────────────

PairwiseSimilarityFn = Callable[
    [str, str, dict[str, set[str]], dict[str, float]],
    tuple[float, str | None],
]
"""
Shared signature for resnik_similarity, lin_similarity,
and jiang_conrath_similarity.

    (term_a, term_b, ancestor_sets, ic_values) -> (score, mica_term | None)
"""

# ── MICA cache ────────────────────────────────────────────────────────────────
# Keyed on (term_a, term_b) — valid as long as ancestor_sets and ic_values
# don't change, which is true for the entire batch evaluation run.

_mica_cache: dict[tuple[str, str], tuple[str | None, float]] = {}


def clear_mica_cache() -> None:
    """
    Clear the MICA cache.

    Call this between patients if memory usage becomes a concern.
    Not needed for normal batch runs — the cache only grows to the size
    of unique (term_a, term_b) pairs seen, which is bounded by the
    HPO vocabulary size.
    """
    _mica_cache.clear()


def cache_stats() -> dict:
    """Return current cache size for debugging."""
    return {"mica_cache_size": len(_mica_cache)}


# ── Ancestor utilities ────────────────────────────────────────────────────────


def get_common_ancestors(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
) -> set[str]:
    """
    Returns the set of shared ancestors between two HPO terms.

    Important:
    - Uses *inclusive* ancestors → includes the term itself
    - Intersection defines the semantic overlap

    Intuition:
    If two terms share many ancestors (especially specific ones),
    they are semantically related.
    """
    ancestors_a = get_ancestors_inclusive(term_a, ancestor_sets)
    ancestors_b = get_ancestors_inclusive(term_b, ancestor_sets)
    return ancestors_a & ancestors_b


def get_mica(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
) -> tuple[str | None, float]:
    """
    Computes the MICA (Most Informative Common Ancestor).

    Definition:
    - Among all shared ancestors, pick the one with highest IC

    Why:
    - IC measures specificity (rare = high IC)
    - The most specific shared ancestor captures the strongest semantic relation

    Returns:
    - mica_term: the ancestor term
    - mica_ic: its IC value

    Performance:
    - Results are cached in _mica_cache keyed on (term_a, term_b).
    - Subsequent calls with the same pair return instantly.
    """
    key = (term_a, term_b)
    if key in _mica_cache:
        return _mica_cache[key]

    # Also check reversed pair — MICA is symmetric
    key_rev = (term_b, term_a)
    if key_rev in _mica_cache:
        return _mica_cache[key_rev]

    common = get_common_ancestors(term_a, term_b, ancestor_sets)
    if not common:
        result = (None, 0.0)
    else:
        mica_term = max(common, key=lambda t: ic_values.get(t, 0.0))
        mica_ic = ic_values.get(mica_term, 0.0)
        result = (mica_term, mica_ic)

    _mica_cache[key] = result
    return result


# ── Similarity functions ──────────────────────────────────────────────────────


def resnik_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
) -> tuple[float, str | None]:
    """
    Resnik similarity.

    Definition:
    similarity = IC(MICA)

    Interpretation:
    - Only considers the shared ancestor (not individual terms)
    - Higher IC → more specific → more similar

    Limitation:
    - Ignores how far terms are from MICA
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    return mica_ic, mica_term


def lin_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
) -> tuple[float, str | None]:
    """
    Lin similarity.

    Definition:
    (2 * IC(MICA)) / (IC(term_a) + IC(term_b))

    Interpretation:
    - Normalizes Resnik by term specificity
    - Produces values in [0, 1]

    Behavior:
    - High when both terms are specific AND share a strong ancestor
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    if mica_term is None:
        return 0.0, None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    denom = ic_a + ic_b

    if denom == 0.0:
        return 0.0, mica_term

    score = (2.0 * mica_ic) / denom
    return score, mica_term


def jiang_conrath_distance(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
) -> tuple[float, str | None]:
    """
    Jiang-Conrath distance.

    Definition:
    distance = IC(a) + IC(b) - 2 * IC(MICA)

    Interpretation:
    - Measures dissimilarity instead of similarity
    - Lower distance = more similar
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    if mica_term is None:
        return float("inf"), None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    distance = ic_a + ic_b - 2.0 * mica_ic
    return max(distance, 0.0), mica_term


def jiang_conrath_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
) -> tuple[float, str | None]:
    """
    Converts Jiang-Conrath distance into similarity.

    Definition:
    similarity = 1 / (1 + distance)

    Interpretation:
    - Bounded in (0,1]
    - Higher value → more similar
    """
    distance, mica_term = jiang_conrath_distance(
        term_a,
        term_b,
        ancestor_sets,
        ic_values,
    )
    if math.isinf(distance):
        return 0.0, None

    score = 1.0 / (1.0 + distance)
    return score, mica_term


# ── BMA aggregation ──────────────────────────────────────────────────────


def best_match_scores(
    source_terms: set[str],
    target_terms: set[str],
    ancestor_sets: dict[str, set[str]],
    ic_values: dict[str, float],
    similarity_fn: PairwiseSimilarityFn,
) -> tuple[float, list[dict]]:
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
