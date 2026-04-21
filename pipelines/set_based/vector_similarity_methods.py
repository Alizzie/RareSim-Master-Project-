"""
Module for vector similarity methods between HPO terms and term sets.
"""

from typing import Dict
import math
import numpy as np


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
