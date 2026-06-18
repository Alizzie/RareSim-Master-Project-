"""
HPO/ontology preprocessing for the semantic similarity pipeline.

Used by:
    similarity_methods/semantic/pipeline.py
"""


def filter_terms_by_ic(
    terms: set[str],
    ic_values: dict[str, float],
    ic_threshold: float | None,
) -> set[str]:
    """
    Filter a set of HPO terms by minimum IC value.

    Removes overly broad terms (low IC) that add noise to similarity scores.
    Example: HP:0000118 (Phenotypic abnormality) has very low IC and is excluded.

    Args:
        terms:        Set of HPO term IDs.
        ic_values:    Dict mapping HPO ID → IC value.
        ic_threshold: Minimum IC to keep a term. None = keep all.
    """
    if ic_threshold is None:
        return set(terms)
    return {term for term in terms if ic_values.get(term, 0.0) >= ic_threshold}


def preprocess_ancestor_sets(
    ancestors: dict[str, list[str]],
) -> dict[str, set[str]]:
    """
    Convert ancestor lists to inclusive ancestor sets.

    Inclusive means the term itself is included in its own ancestor set.
    Precomputed once and reused across all BMA comparisons for efficiency.

    Args:
        ancestors: Dict mapping HPO ID → list of ancestor IDs.

    Returns:
        Dict mapping HPO ID → set of ancestor IDs (including self).
    """
    return {
        term: set(parent_terms) | {term} for term, parent_terms in ancestors.items()
    }


def get_ancestors_inclusive(
    term: str,
    ancestor_sets: dict[str, set[str]],
) -> set[str]:
    """Return the inclusive ancestor set for a term (includes the term itself)."""
    return ancestor_sets.get(term, {term})
