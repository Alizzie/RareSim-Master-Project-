"""
Core math functions for similarity calculations.
All functions operate on dicts (sparse vectors) as the canonical input type.
Sets are converted to binary vectors before computation.
"""

import math
from typing import Callable, Dict, List, Optional, Set, Union, Tuple

TermSet = Set[str]
TermVector = Dict[str, float]
TermInput = Union[TermSet, TermVector]
PairwiseSimilarityFn = Callable[
    [str, str, Dict[str, Set[str]], Dict[str, float]],
    Tuple[float, Optional[str]],
]


def to_binary_vector(terms: TermInput) -> TermVector:
    """
    Convert a set of terms to a binary vector representation.
    If input is already a vector, it is returned as-is.
    """
    if isinstance(terms, dict):
        return terms
    elif isinstance(terms, set):
        return {term: 1.0 for term in terms}
    else:
        raise ValueError("Input must be a set of terms or a term vector.")


def cosine_similarity(
    vec_a: TermInput, vec_b: TermInput, use_binary: bool = False
) -> float:
    """
    Cosine similarity between two sparse vectors.
    Accepts sets (converted to binary vectors) or dicts.
    Returns a value in [0, 1] where 1 = identical profile.
    Formula: cos(A, B) = (A * B) / (||A|| x ||B||)
    """

    a = to_binary_vector(vec_a) if use_binary or isinstance(vec_a, set) else vec_a
    b = to_binary_vector(vec_b) if use_binary or isinstance(vec_b, set) else vec_b

    if not a or not b:
        return 0.0

    dot = sum(a[t] * b[t] for t in a if t in b)

    a_norm = math.sqrt(sum(v**2 for v in a.values()))
    b_norm = math.sqrt(sum(v**2 for v in b.values()))

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return dot / (a_norm * b_norm)


def jaccard(vec_a: TermInput, vec_b: TermInput) -> float:
    """
    Jaccard similarity between two sets or binary vectors.
    |intersection| / |union|
    Returns a value in [0, 1] where 1 = identical sets.
    """
    a = set(to_binary_vector(vec_a).keys())
    b = set(to_binary_vector(vec_b).keys())

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def dice(vec_a: TermInput, vec_b: TermInput) -> float:
    """
    Dice similarity (Sørensen–Dice) between two sets or binary vectors.
    2 * |intersection| / (|A| + |B|)
    Returns a value in [0, 1] where 1 = identical sets.
    """
    a = set(to_binary_vector(vec_a).keys())
    b = set(to_binary_vector(vec_b).keys())

    if not a or not b:
        return 0.0

    return 2 * len(a & b) / (len(a) + len(b))


def overlap_coefficient(vec_a: TermInput, vec_b: TermInput) -> float:
    """
    Overlap coefficient (Szymkiewicz-Simpson) between two sets or binary vectors.
    |intersection| / min(|A|, |B|)
    Returns a value in [0, 1] where 1 = one set is a subset of the other.
    """
    a = set(to_binary_vector(vec_a).keys())
    b = set(to_binary_vector(vec_b).keys())

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def filter_terms_by_ic(
    terms: Set[str],
    ic_values: Dict[str, float],
    ic_threshold: Optional[float],
) -> Set[str]:
    """
    Filter a set of HPO terms by minimum IC value.

    Removes overly broad terms (low IC) that add noise to similarity scores.
    Example: HP:0000118 (Phenotypic abnormality) has very low IC and is excluded.

    Args:
        terms:        Set of HPO term IDs.
        ic_values:    Dict mapping HPO ID → IC value.
        ic_threshold: Minimum IC to keep a term. None = keep all.
    """
    if ic_threshold is None:
        return set(terms)
    return {term for term in terms if ic_values.get(term, 0.0) >= ic_threshold}


def sum_ic(terms: Set[str], ic_values: Dict[str, float]) -> float:
    """Sum IC values for a set of HPO terms."""
    return sum(ic_values.get(term, 0.0) for term in terms)


def preprocess_ancestor_sets(
    ancestors: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    """
    Convert ancestor lists to inclusive ancestor sets.

    Inclusive means the term itself is included in its own ancestor set.
    Precomputed once and reused across all BMA comparisons for efficiency.

    Args:
        ancestors: Dict mapping HPO ID → list of ancestor IDs.

    Returns:
        Dict mapping HPO ID → set of ancestor IDs (including self).
    """
    return {
        term: set(parent_terms) | {term}
        for term, parent_terms in ancestors.items()
    }


def get_ancestors_inclusive(
    term: str,
    ancestor_sets: Dict[str, Set[str]],
) -> Set[str]:
    """Return the inclusive ancestor set for a term (includes the term itself)."""
    return ancestor_sets.get(term, {term})
