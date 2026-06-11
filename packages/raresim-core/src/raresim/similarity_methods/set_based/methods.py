"""Set-based similarity methods for comparing patient and disease term vectors."""

from typing import Tuple
from shared.math import (
    TermInput,
    jaccard as _jaccard,
    dice as _dice,
    overlap_coefficient as _overlap,
    to_binary_vector,
)
from shared.methods import _empty_explanation


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
