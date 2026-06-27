"""
HPO/ontology preprocessing for the semantic similarity pipeline.

Used by:
    similarity_methods/semantic/pipeline.py
"""


def get_hpo_label(term: str, hpo_labels: dict[str, str]) -> str:
    """Return the label for one HPO term, falling back to the term ID."""
    return hpo_labels.get(term) or term


def convert_hpo_ids_to_labels(
    hpo_terms: set[str],
    hpo_labels: dict[str, str],
    *,
    skip_unlabeled: bool = True,
) -> list[str]:
    """
    Resolve a list of HPO IDs to label strings, deduplicated, order-preserved.

    skip_unlabeled=True drops terms with no label (transformer embedding text —
    you don't want raw HP:IDs in the embedded string). skip_unlabeled=False
    keeps the ID as fallback via get_hpo_label (prompt display, where showing
    the ID is fine).
    """
    seen: set[str] = set()
    out: list[str] = []
    for term in hpo_terms:
        label = hpo_labels.get(term)
        if not label:
            if skip_unlabeled:
                continue
            label = term
        label = label.strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


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
