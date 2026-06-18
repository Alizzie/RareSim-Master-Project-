"""
Reusable similarity methods with explanations.
All methods wrap the pure math from utils.math and add an explanation dict.
Accepts both sets and dicts (binary vectors) as input.

Used by:
    similarity_methods/set_based/methods.py
"""

from raresim.utils.similarity_math import (
    TermInput,
    cosine_similarity as _cosine,
)


def cosine_similarity(
    pat: TermInput, disease: TermInput, use_binary: bool = False
) -> float:
    """
    Cosine similarity between two vectors.
    """
    if not pat or not disease:
        return 0.0

    score = _cosine(pat, disease, use_binary)
    return score
