"""
core/disease_ancestors.py — Build ORDO ancestor chains for ontological distance.

Mirrors the HPO ancestor pipeline:
    load_hpo_owl      →  load_ordo_parents   (in core/loaders.py)
    compute_ancestors →  reused directly

Output: ORPHA:NNNN → [root, ..., immediate_parent]  (ordered, root first)
"""

from core.hpo_utils import compute_ancestors


def build_ordered_ancestor_chains(
    disease_parents: dict[str, set[str]],
) -> dict[str, list[str]]:
    """
    Convert a child→{parents} map into ordered ancestor chains.

    Uses compute_ancestors to get full ancestor sets, then traces a single
    deterministic root-to-parent path by always stepping to the deepest
    available parent (smallest ancestor set = most specific).

    Returns:
        ORPHA:NNNN → [root, ..., immediate_parent]
        Root nodes (no parents) map to an empty list.
    """
    ancestor_sets = compute_ancestors(disease_parents)

    return {
        node: _trace_path_to_root(node, disease_parents, ancestor_sets)
        for node in disease_parents
    }


def _trace_path_to_root(
    node: str,
    disease_parents: dict[str, set[str]],
    ancestor_sets: dict[str, set[str]],
) -> list[str]:
    """
    Walk upward from node, picking the deepest parent at each step.

    In ORDO's DAG a disease can have multiple IS-A parents (e.g. classified
    under both a clinical and an aetiological grouping).  Choosing the parent
    with the smallest ancestor set gives the most specific single path, which
    makes the LCA in compare_methods.py as tight as possible.

    Returns [root, ..., immediate_parent], NOT including node itself.
    """
    chain: list[str] = []
    current = node
    visited: set[str] = {node}

    while True:
        parents = disease_parents.get(current, set())
        if not parents:
            break

        # Deepest parent = fewest ancestors above it
        best = min(parents, key=lambda p: len(ancestor_sets.get(p, set())))

        if best in visited:  # cycle guard (shouldn't occur in ORDO)
            break

        chain.append(best)
        visited.add(best)
        current = best

    chain.reverse()  # root → immediate parent
    return chain
