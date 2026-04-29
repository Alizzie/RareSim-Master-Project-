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


def jaccard_similarity(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Jaccard similarity between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("jaccard", pat)

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _jaccard(pat, disease)
    explanation = {
        "method": "jaccard",
        "intersection_size": len(pat & disease),
        "union_size": len(pat | disease),
        "top_shared_terms": sorted(pat & disease)[:10],
    }
    return score, explanation


def dice_similarity(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Dice similarity between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("dice", pat)

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _dice(pat, disease)
    explanation = {
        "method": "dice",
        "intersection_size": len(pat & disease),
        "size_patient": len(pat),
        "size_disease": len(disease),
        "top_shared_terms": sorted(pat & disease)[:10],
    }
    return score, explanation


def overlap_coefficient(pat: TermInput, disease: TermInput) -> Tuple[float, dict]:
    """Calculate Overlap Coefficient between two vectors with an explanation."""
    if not pat or not disease:
        return 0.0, _empty_explanation("overlap_coefficient", pat)

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _overlap(pat, disease)
    explanation = {
        "method": "overlap_coefficient",
        "intersection_size": len(pat & disease),
        "min_size": min(len(pat), len(disease)),
        "top_shared_terms": sorted(pat & disease)[:10],
    }
    return score, explanation
