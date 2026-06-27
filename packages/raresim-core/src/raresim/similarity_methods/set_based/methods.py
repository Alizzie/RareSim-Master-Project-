"""Set-based similarity methods for comparing patient and disease term vectors."""

from raresim.utils.similarity_math import (
    TermInput,
    jaccard as _jaccard,
    dice as _dice,
    overlap_coefficient as _overlap,
    cosine_similarity as _cosine,
    to_binary_vector,
)


def cosine_similarity(pat: TermInput, disease: TermInput) -> float:
    """
    Cosine similarity between two vectors.
    """
    if not pat or not disease:
        return 0.0

    score = _cosine(pat, disease, False)
    return score


def jaccard_similarity(pat: TermInput, disease: TermInput) -> float:
    """Calculate Jaccard similarity between two vectors"""
    if not pat or not disease:
        return 0.0

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _jaccard(pat, disease)
    return score


def dice_similarity(pat: TermInput, disease: TermInput) -> float:
    """Calculate Dice similarity between two vectors"""
    if not pat or not disease:
        return 0.0

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _dice(pat, disease)
    return score


def overlap_coefficient(pat: TermInput, disease: TermInput) -> float:
    """Calculate Overlap Coefficient between two vectors"""
    if not pat or not disease:
        return 0.0

    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    score = _overlap(pat, disease)
    return score
