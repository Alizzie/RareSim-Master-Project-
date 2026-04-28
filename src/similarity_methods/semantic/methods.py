"""
Module for semantic similarity methods between HPO terms and term sets.

Core idea:
- Terms are compared using the ontology structure (ancestors)
- Information Content (IC) is used to measure specificity
- More specific shared ancestors → higher similarity
"""

import math
from typing import Dict, Optional, Set, Tuple

from .utils import get_ancestors_inclusive, sum_ic


def get_common_ancestors(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
) -> Set[str]:
    """
    Returns the set of shared ancestors between two HPO terms.

    Important:
    - Uses *inclusive* ancestors → includes the term itself
    - Intersection defines the semantic overlap

    Intuition:
    If two terms share many ancestors (especially specific ones),
    they are semantically related.
    """
    ancestors_a = get_ancestors_inclusive(term_a, ancestor_sets)
    ancestors_b = get_ancestors_inclusive(term_b, ancestor_sets)
    return ancestors_a & ancestors_b


def get_mica(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[Optional[str], float]:
    """
    Computes the MICA (Most Informative Common Ancestor).

    Definition:
    - Among all shared ancestors, pick the one with highest IC

    Why:
    - IC measures specificity (rare = high IC)
    - The most specific shared ancestor captures the strongest semantic relation

    Returns:
    - mica_term: the ancestor term
    - mica_ic: its IC value
    """
    common = get_common_ancestors(term_a, term_b, ancestor_sets)
    if not common:
        return None, 0.0

    mica_term = max(common, key=lambda t: ic_values.get(t, 0.0))
    mica_ic = ic_values.get(mica_term, 0.0)
    return mica_term, mica_ic


def resnik_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Resnik similarity.

    Definition:
    similarity = IC(MICA)

    Interpretation:
    - Only considers the shared ancestor (not individual terms)
    - Higher IC → more specific → more similar

    Limitation:
    - Ignores how far terms are from MICA
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    return mica_ic, mica_term


def lin_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Lin similarity.

    Definition:
    (2 * IC(MICA)) / (IC(term_a) + IC(term_b))

    Interpretation:
    - Normalizes Resnik by term specificity
    - Produces values in [0, 1]

    Behavior:
    - High when both terms are specific AND share a strong ancestor
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    if mica_term is None:
        return 0.0, None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    denom = ic_a + ic_b

    if denom == 0.0:
        return 0.0, mica_term

    score = (2.0 * mica_ic) / denom
    return score, mica_term


def jiang_conrath_distance(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Jiang-Conrath distance.

    Definition:
    distance = IC(a) + IC(b) - 2 * IC(MICA)

    Interpretation:
    - Measures dissimilarity instead of similarity
    - Lower distance = more similar
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestor_sets, ic_values)
    if mica_term is None:
        return float("inf"), None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    distance = ic_a + ic_b - 2.0 * mica_ic
    return max(distance, 0.0), mica_term


def jiang_conrath_similarity(
    term_a: str,
    term_b: str,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Converts Jiang-Conrath distance into similarity.

    Definition:
    similarity = 1 / (1 + distance)

    Interpretation:
    - Bounded in (0,1]
    - Higher value → more similar
    """
    distance, mica_term = jiang_conrath_distance(
        term_a,
        term_b,
        ancestor_sets,
        ic_values,
    )
    if math.isinf(distance):
        return 0.0, None

    score = 1.0 / (1.0 + distance)
    return score, mica_term


def simgic_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: Dict[str, float],
) -> Tuple[float, dict]:
    """
    simGIC similarity (set-based, IC-weighted Jaccard).

    Definition:
    sum(IC(intersection)) / sum(IC(union))

    Interpretation:
    - Extends Jaccard by weighting terms by importance (IC)
    - Rewards matching rare/specific phenotypes more

    Output:
    - similarity score
    - explanation with statistics
    """
    if not patient_terms or not disease_terms:
        return 0.0, {
            "intersection_size": 0,
            "union_size": 0,
            "intersection_ic_sum": 0.0,
            "union_ic_sum": 0.0,
        }

    intersection = patient_terms & disease_terms
    union = patient_terms | disease_terms

    intersection_ic = sum_ic(intersection, ic_values)
    union_ic = sum_ic(union, ic_values)

    score = 0.0 if union_ic == 0.0 else intersection_ic / union_ic

    explanation = {
        "intersection_size": len(intersection),
        "union_size": len(union),
        "intersection_ic_sum": intersection_ic,
        "union_ic_sum": union_ic,
        "top_shared_terms": sorted(
            intersection,
            key=lambda t: ic_values.get(t, 0.0),
            reverse=True,
        )[:10],
    }
    return score, explanation


def icto_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: Dict[str, float],
) -> Tuple[float, dict]:
    """
    ICTO similarity (IC-weighted overlap coefficient).

    Definition:
    sum(IC(intersection)) / min(sum(IC(patient)), sum(IC(disease)))

    Interpretation:
    - Measures how much the smaller set is covered
    - More sensitive than simGIC when sizes differ

    Use case:
    - Good when patient has few symptoms but matches disease well
    """
    if not patient_terms or not disease_terms:
        return 0.0, {
            "intersection_size": 0,
            "patient_ic_sum": 0.0,
            "disease_ic_sum": 0.0,
            "intersection_ic_sum": 0.0,
        }

    intersection = patient_terms & disease_terms
    patient_ic = sum_ic(patient_terms, ic_values)
    disease_ic = sum_ic(disease_terms, ic_values)
    intersection_ic = sum_ic(intersection, ic_values)

    denom = min(patient_ic, disease_ic)
    score = 0.0 if denom == 0.0 else intersection_ic / denom

    explanation = {
        "intersection_size": len(intersection),
        "patient_ic_sum": patient_ic,
        "disease_ic_sum": disease_ic,
        "intersection_ic_sum": intersection_ic,
        "top_shared_terms": sorted(
            intersection,
            key=lambda t: ic_values.get(t, 0.0),
            reverse=True,
        )[:10],
    }
    return score, explanation


def jaccard_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: Dict[str, float],
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
