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
    """Cosine similarity between two vectors."""
    if not pat or not disease:
        return 0.0
    return _cosine(pat, disease, False)


def jaccard_similarity(pat: TermInput, disease: TermInput) -> float:
    """Calculate Jaccard similarity between two vectors"""
    if not pat or not disease:
        return 0.0
    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    return _jaccard(pat, disease)


def dice_similarity(pat: TermInput, disease: TermInput) -> float:
    """Calculate Dice similarity between two vectors"""
    if not pat or not disease:
        return 0.0
    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    return _dice(pat, disease)


def overlap_coefficient(pat: TermInput, disease: TermInput) -> float:
    """Calculate Overlap Coefficient between two vectors"""
    if not pat or not disease:
        return 0.0
    pat = set(to_binary_vector(pat).keys())
    disease = set(to_binary_vector(disease).keys())
    return _overlap(pat, disease)


def jaccard_with_negative_penalty(
    pat: TermInput,
    disease: TermInput,
    pat_excluded: TermInput | None = None,
    disease_excluded: TermInput | None = None,
    penalty_weight: float = 0.5,
) -> float:
    """
    Jaccard similarity penalized by contradicting evidence from excluded terms.

    Two kinds of contradiction are penalized:
      1. The patient was explicitly tested negative for a term that is part
         of the disease's profile (pat_excluded ∩ disease).
      2. The disease explicitly excludes a term the patient has
         (disease_excluded ∩ pat).

    score = jaccard(pat, disease) - penalty_weight * (n_contradictions / union_size)
    Result is clamped to [0, 1].
    """
    if not pat or not disease:
        return 0.0

    pat_set = set(to_binary_vector(pat).keys())
    disease_set = set(to_binary_vector(disease).keys())

    base_score = _jaccard(pat_set, disease_set)

    pat_excluded = set(pat_excluded or [])
    disease_excluded = set(disease_excluded or [])

    contradictions = (pat_excluded & disease_set) | (disease_excluded & pat_set)
    if not contradictions:
        return base_score

    union_size = len(pat_set | disease_set)
    if union_size == 0:
        return base_score

    penalty = penalty_weight * (len(contradictions) / union_size)
    return max(0.0, base_score - penalty)
