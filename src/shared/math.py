"""
Core math functions for similarity calculations.
All functions operate on dicts (sparse vectors) as the canonical input type.
Sets are converted to binary vectors before computation.
"""

import math
from typing import Dict, Set, Union

TermSet = Set[str]
TermVector = Dict[str, float]
TermInput = Union[TermSet, TermVector]


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
