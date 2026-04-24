"""
Module for vector similarity methods between HPO terms and term sets.
"""

from typing import Dict, Set, Tuple
import math


def overlap_coefficient(pat_set: set, disease_set: set) -> float:
    """
    Overlap coefficient (Szymkiewicz–Simpson coefficient) between two sets.
    Formula: overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    Returns a value in [0, 1] where 1 = one set is a subset of the other.
    """

    explaination = {}
    explaination["method"] = "overlap_coefficient"
    explaination["description"] = (
        "Overlap coefficient between two sets, measuring the size of the intersection relative to the smaller set."
    )

    if not pat_set or not disease_set:
        explaination["failure_reason"] = (
            f"{ 'Patient' if not pat_set else 'Disease' } set is empty, resulting in zero similarity."
        )
        return 0.0, explaination

    return (
        len(pat_set & disease_set) / min(len(pat_set), len(disease_set)),
        explaination,
    )


def jaccard_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: None,
) -> Tuple[float, dict]:
    """
    Standard Jaccard similarity (baseline).

    Definition:
    |intersection| / |union|

    Interpretation:
    - Treats all terms equally (no IC weighting)
    - Useful baseline for comparison

    Note:
    - Ignores ontology structure and term importance
    """
    if ic_values is not None:
        del ic_values

    if not patient_terms or not disease_terms:
        return 0.0, {
            "intersection_size": 0,
            "union_size": 0,
            "top_shared_terms": [],
        }

    intersection = patient_terms & disease_terms
    union = patient_terms | disease_terms
    score = 0.0 if not union else len(intersection) / len(union)

    explanation = {
        "intersection_size": len(intersection),
        "union_size": len(union),
        "top_shared_terms": sorted(intersection)[:10],
    }
    return score, explanation


def dice(pat_set: set, disease_set: set) -> float:
    """
    Dice similarity (Sørensen–Dice coefficient) between two sets.
    Formula: Dice(A, B) = 2 * |A ∩ B| / (|A| + |B|)
    Returns a value in [0, 1] where 1 = identical sets.
    """
    explaination = {}
    explaination["method"] = "dice"
    explaination["description"] = (
        "Dice similarity between two sets, measuring the size of the intersection relative to the average size of the two sets."
    )

    if not pat_set or not disease_set:
        explaination["failure_reason"] = (
            f"{ 'Patient' if not pat_set else 'Disease' } set is empty, resulting in zero similarity."
        )
        return 0.0, explaination

    return (
        2 * len(pat_set & disease_set) / (len(pat_set) + len(disease_set)),
        explaination,
    )


def cosine_similarity(
    pat_vec: Dict[str, float],
    disease_vec: Dict[str, float],
) -> float:
    """
    Cosine similarity between two vectors (e.g. patient vector against disease vector) suitable for sparse representations.
    Formula: cos(A, B) = (A * B) / (||A|| x ||B||)
    Returns a value in [0, 1] where 1 = identical term profile.
    """

    explaination = {}
    explaination["method"] = "cosine_similarity"
    explaination["description"] = (
        "Cosine similarity between two vectors suitable for sparse representations."
    )

    if not pat_vec or not disease_vec:
        explaination["failure_reason"] = (
            f"{ 'Patient' if not pat_vec else 'Disease' } vector is empty, resulting in zero similarity."
        )
        return 0.0, explaination

    dot = sum(pat_vec[t] * disease_vec[t] for t in pat_vec if t in disease_vec)

    pat_norm = math.sqrt(sum(v**2 for v in pat_vec.values()))
    disease_norm = math.sqrt(sum(v**2 for v in disease_vec.values()))

    if pat_norm == 0.0 or disease_norm == 0.0:
        explaination["failure_reason"] = (
            f"{ 'Patient' if pat_norm == 0.0 else 'Disease' } vector has zero magnitude, resulting in zero similarity."
        )
        return 0.0, explaination

    explaination["dot_product"] = dot
    explaination["norm_patient"] = pat_norm
    explaination["norm_disease"] = disease_norm

    return dot / (pat_norm * disease_norm), explaination
