"""
Reusable similarity methods with explanations.
All methods wrap the pure math from shared.math and add an explanation dict.
Accepts both sets and dicts (binary vectors) as input.
"""

from typing import Tuple
from shared.math import (
    TermInput,
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
