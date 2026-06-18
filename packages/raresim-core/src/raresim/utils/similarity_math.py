"""
Pure mathematical primitives for similarity calculations.

All functions operate on dicts (sparse vectors) as the canonical input type.
Sets are converted to binary vectors before computation.

Type aliases
------------
TermSet    : set of string term IDs
TermVector : sparse float vector keyed by term ID
TermInput  : either of the above (converted internally as needed)
"""

import math
import numpy as np

TermSet = set[str]
TermVector = dict[str, float]
TermInput = TermSet | TermVector


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

    Args:
        vec_a, vec_b : term vectors or sets.
        use_binary   : if True, convert both inputs to binary before scoring.
                       Useful when you have weighted vectors but want a
                       binary comparison.
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


def cosine_similarity_dense(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two dense numpy vectors.

    Kept here so HPO2Vec and Autoencoder pipelines share one implementation
    instead of each defining their own. Import numpy lazily so the rest of
    utils/math.py has no numpy dependency.
    """

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


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


def sum_ic(terms: set[str], ic_values: dict[str, float]) -> float:
    """Sum IC values for a set of HPO terms."""
    return sum(ic_values.get(term, 0.0) for term in terms)
