from typing import Dict, Set

"""Functions to compute ancestors of HPO terms and apply the true-path rule for term propagation."""


def compute_ancestors(hpo_parents: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Compute all ancestors for every HPO term using DFS.
    """
    cache: Dict[str, Set[str]] = {}

    def dfs(term: str) -> Set[str]:
        if term in cache:
            return cache[term]

        ancestors: Set[str] = set()
        for parent in hpo_parents.get(term, set()):
            ancestors.add(parent)
            ancestors.update(dfs(parent))

        cache[term] = ancestors
        return ancestors

    for term in hpo_parents:
        dfs(term)

    return cache


def compute_parents(hpo_parents: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Convert parent sets to sorted lists for easier serialization."""

    return {k: sorted(v) for k, v in hpo_parents.items()}


def propagate_hpo_terms(
    terms: Set[str], hpo_ancestors: Dict[str, Set[str]]
) -> Set[str]:
    """
    Apply true-path rule: add all ancestor HPO terms.
    """
    propagated = set(terms)
    for term in terms:
        propagated.update(hpo_ancestors.get(term, set()))
    return propagated
