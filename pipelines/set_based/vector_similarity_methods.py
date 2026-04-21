"""
Module for vector similarity methods between HPO terms and term sets.
"""

from typing import Dict
import math
import numpy as np


def cosine_similarity_sparse(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float],
) -> float:
    """
    Cosine similarity between two vectors (e.g. patient vector against disease vector).
    Formula: cos(A, B) = (A * B) / (||A|| x ||B||)
    Returns a value in [0, 1] where 1 = identical term profile.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(vec_a[t] * vec_b[t] for t in vec_a if t in vec_b)

    norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def cosine_similarity_dense(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
