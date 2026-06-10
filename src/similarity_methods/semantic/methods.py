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
from typing import Dict, Optional, Set, Tuple

from shared.math import get_ancestors_inclusive

# ── MICA cache ────────────────────────────────────────────────────────────────
# Keyed on (term_a, term_b) — valid as long as ancestor_sets and ic_values
# don't change, which is true for the entire batch evaluation run.

_mica_cache: dict[tuple[str, str], tuple[Optional[str], float]] = {}


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
    ancestor_sets: Dict[str, Set[str]],
) -> Set[str]:
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
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[Optional[str], float]:
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
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
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
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
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
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
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
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
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
    