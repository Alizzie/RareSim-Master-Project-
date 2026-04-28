"""
Reusable similarity methods with explanations.
All methods wrap the pure math from shared.math and add an explanation dict.
Accepts both sets and dicts (binary vectors) as input.
"""

from typing import Tuple
from shared.math import (
    TermInput,
    cosine_similarity as _cosine,
    jaccard as _jaccard,
    dice as _dice,
    overlap_coefficient as _overlap,
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
    return sorted(a & b)[:10]


def cosine_similarity(
    pat: TermInput, disease: TermInput, use_binary: bool = False
) -> Tuple[float, dict]:
    """
    Cosine similarity between two vectors with an explanation of the calculation steps.
    """
    if not pat or not disease:
        return 0.0, _empty_explanation("cosine_similarity", pat)

    score = _cosine(pat, disease, use_binary)
    explanation = {
        "method": "cosine_similarity",
        "score": score,
        "top_shared_terms": _shared_terms(pat, disease),
    }
    return score, explanation


def jaccard_similarity(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Jaccard similarity between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("jaccard", pat)

    a = set(to_binary_vector(pat).keys())
    b = set(to_binary_vector(disease).keys())
    score = _jaccard(pat, disease)
    explanation = {
        "method": "jaccard",
        "intersection_size": len(a & b),
        "union_size": len(a | b),
        "top_shared_terms": sorted(a & b)[:10],
    }
    return score, explanation


def dice_similarity(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Dice similarity between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("dice", pat)

    a = set(to_binary_vector(pat).keys())
    b = set(to_binary_vector(disease).keys())
    score = _dice(pat, disease)
    explanation = {
        "method": "dice",
        "intersection_size": len(a & b),
        "size_a": len(a),
        "size_b": len(b),
        "top_shared_terms": sorted(a & b)[:10],
    }
    return score, explanation


def overlap_coefficient(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Overlap Coefficient between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("overlap_coefficient", pat)

    a = set(to_binary_vector(pat).keys())
    b = set(to_binary_vector(disease).keys())
    score = _overlap(pat, disease)
    explanation = {
        "method": "overlap_coefficient",
        "intersection_size": len(a & b),
        "min_size": min(len(a), len(b)),
        "top_shared_terms": sorted(a & b)[:10],
    }
    return score, explanation
