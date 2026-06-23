"""
RareSim method-comparison core logic.

This module computes single-case method agreement data from per-method ranked
disease lists.

The output describes agreement, consensus, and confidence across methods.
It does not claim correctness, because ground truth is unavailable at inference
time.
"""

from collections import defaultdict
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # pylint: disable=invalid-name


RRF_K = 60

SHORT_NAMES = {
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
    "emilyalsentzer/Bio_ClinicalBERT": "Bio_ClinicalBERT",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "SapBERT",
    "dmis-lab/biobert-v1.1": "BioBERT",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "semantic_resnik_bma": "Resnik BMA",
    "semantic_lin_bma": "Lin BMA",
    "semantic_jiang_conrath_bma": "JiangConrath BMA",
    "denoising_autoencoder": "Autoencoder",
    "tfidf_cosine": "TF-IDF",
    "tfidf": "TF-IDF",
    "hpo2vec": "HPO2Vec",
}

CATEGORY_ORDER = [
    "set",
    "semantic",
    "embedding",
    "tfidf",
    "hpo2vec",
    "autoencoder",
    "llm",
]

CATEGORY_LABELS = {
    "set": "set-based",
    "semantic": "semantic",
    "embedding": "embedding",
    "tfidf": "tf-idf",
    "hpo2vec": "hpo2vec",
    "autoencoder": "autoencoder",
    "llm": "llm",
}


def short_name(method: str) -> str:
    """Return a compact display name for a method identifier."""
    if method in SHORT_NAMES:
        return SHORT_NAMES[method]

    tail = method.split("transformer_", 1)[-1]

    if tail in SHORT_NAMES:
        return SHORT_NAMES[tail]

    if "/" in tail:
        return tail.split("/")[-1]

    return tail


def method_category(method: str) -> str:
    """Return the high-level method category."""
    method_lower = method.lower()

    category = "embedding"

    if method.startswith("set_"):
        category = "set"
    elif method.startswith("semantic_"):
        category = "semantic"
    elif "tfidf" in method:
        category = "tfidf"
    elif method == "hpo2vec":
        category = "hpo2vec"
    elif method == "denoising_autoencoder":
        category = "autoencoder"
    elif "mistral" in method_lower or method.endswith("ordo_llm") or method == "llm":
        category = "llm"

    return category


def _category_sort_key(category: str) -> int:
    """Return the configured sort index for a method category."""
    if category in CATEGORY_ORDER:
        return CATEGORY_ORDER.index(category)

    return len(CATEGORY_ORDER)


def _disease_id(item: dict[str, Any]) -> str | None:
    """Extract the disease identifier from a heterogeneous result item."""
    disease_id = (
        item.get("disease_id")
        or item.get("canonical_disease_id")
        or item.get("ordo_id")
    )

    if not disease_id:
        return None

    return str(disease_id)


def _label(item: dict[str, Any]) -> str:
    """Extract the disease label from a heterogeneous result item."""
    return str(item.get("label") or item.get("disease_name") or "")


def _score(item: dict[str, Any]) -> float:
    """Extract the numeric score from a result item."""
    try:
        return float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sort_key(item: dict[str, Any]) -> tuple[int, float]:
    """Sort by explicit rank when available, otherwise by descending score."""
    rank = item.get("rank")
    if rank is None:
        return 1, -_score(item)
    try:
        return 0, float(int(rank))
    except (TypeError, ValueError):
        return 1, -_score(item)


def _ranked_topk(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Return top-k disease results with clean per-method ranks."""
    ranked_items = sorted(items, key=_sort_key)[:k]
    output = []

    for index, item in enumerate(ranked_items):
        disease_id = _disease_id(item)

        if not disease_id:
            continue

        label = _label(item) or disease_id

        output.append(
            {
                "disease_id": disease_id,
                "label": label,
                "profile_type": item.get("profile_type"),
                "category_path": item.get("category_path", []),
                "score": round(_score(item), 6),
                "rank": index + 1,
            }
        )

    return output


def normalize_by_method(
    by_method: dict[str, list[dict[str, Any]]],
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize raw per-method results into clean top-k ranked lists."""
    return {
        method: _ranked_topk(items, k)
        for method, items in by_method.items()
        if items
    }


def _collect_fusion_inputs(
    norm: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, float],
    dict[str, set[str]],
    dict[str, dict[str, int]],
    dict[str, str],
]:
    """Collect intermediate values for reciprocal rank fusion."""
    fused = defaultdict(float)
    support = defaultdict(set)
    method_ranks = defaultdict(dict)
    labels = {}

    for method, items in norm.items():
        for item in items:
            disease_id = item["disease_id"]
            rank = item["rank"]

            fused[disease_id] += 1.0 / (RRF_K + rank)
            support[disease_id].add(method)
            method_ranks[disease_id][method] = rank
            labels.setdefault(disease_id, item["label"])

    return fused, support, method_ranks, labels


def _build_fused_row(
    disease_id: str,
    score: float,
    support: dict[str, set[str]],
    method_ranks: dict[str, dict[str, int]],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Build one consensus row for a disease candidate."""
    ranks = list(method_ranks[disease_id].values())
    categories = sorted(
        {method_category(method) for method in support[disease_id]},
        key=_category_sort_key,
    )

    return {
        "disease_id": disease_id,
        "label": labels[disease_id],
        "rrf_score": round(score, 6),
        "n_methods": len(support[disease_id]),
        "methods": sorted(support[disease_id]),
        "n_categories": len(categories),
        "categories": categories,
        "method_ranks": method_ranks[disease_id],
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
        "rank_spread": max(ranks) - min(ranks),
    }


def fuse_rrf(
    norm: dict[str, list[dict[str, Any]]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Fuse per-method rankings with reciprocal rank fusion."""
    fused, support, method_ranks, labels = _collect_fusion_inputs(norm)

    rows = [
        _build_fused_row(disease_id, score, support, method_ranks, labels)
        for disease_id, score in fused.items()
    ]

    rows.sort(key=lambda row: (-row["rrf_score"], row["best_rank"]))

    for index, row in enumerate(rows):
        row["rank"] = index + 1

    return rows[:top_n]


def _seriate(matrix: list[list[float]]) -> list[int]:
    """Order similar methods next to each other using classical MDS."""
    size = len(matrix)

    if np is None or size < 3:
        return list(range(size))

    try:
        similarity = np.array(matrix)
        squared_distance = (1.0 - similarity) ** 2
        centering = np.eye(size) - np.ones((size, size)) / size
        gram = -0.5 * centering @ squared_distance @ centering
        values, vectors = np.linalg.eigh(gram)
        axis = vectors[:, -1] * np.sqrt(max(values[-1], 0.0))

        return [int(index) for index in np.argsort(axis)]
    except (ValueError, TypeError, np.linalg.LinAlgError):  # pragma: no cover
        return list(range(size))


def _jaccard_similarity(first: set[str], second: set[str], same_item: bool) -> float:
    """Compute Jaccard similarity between two disease-id sets."""
    union_size = len(first | second)

    if union_size:
        return len(first & second) / union_size

    if same_item:
        return 1.0

    return 0.0


def agreement_jaccard(norm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute pairwise top-k Jaccard agreement between methods."""
    methods = list(norm)
    disease_sets = {
        method: {item["disease_id"] for item in norm[method]}
        for method in methods
    }

    matrix = []

    for first_index, first_method in enumerate(methods):
        row = []

        for second_index, second_method in enumerate(methods):
            similarity = _jaccard_similarity(
                disease_sets[first_method],
                disease_sets[second_method],
                first_index == second_index,
            )
            row.append(similarity)

        matrix.append(row)

    order = _seriate(matrix)
    ordered_methods = [methods[index] for index in order]
    ordered_matrix = [
        [round(matrix[first][second], 4) for second in order]
        for first in order
    ]

    return {
        "metric": "jaccard",
        "methods_ordered": ordered_methods,
        "categories": {
            method: method_category(method)
            for method in ordered_methods
        },
        "matrix": ordered_matrix,
    }


def _available_categories(norm: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Return method categories present in the normalized result set."""
    present_categories = {method_category(method) for method in norm}

    return [
        category
        for category in CATEGORY_ORDER
        if category in present_categories
    ]


def _best_rank_by_category(
    norm: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Compute the best rank per disease and method category."""
    best = defaultdict(dict)

    for method, items in norm.items():
        category = method_category(method)

        for item in items:
            disease_id = item["disease_id"]
            rank = item["rank"]

            if category not in best[disease_id] or rank < best[disease_id][category]:
                best[disease_id][category] = rank

    return best


def grid_by_category(
    norm: dict[str, list[dict[str, Any]]],
    consensus: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build best-rank grid data for consensus candidates by method category."""
    categories = _available_categories(norm)
    best = _best_rank_by_category(norm)
    rows = []

    for consensus_row in consensus:
        disease_id = consensus_row["disease_id"]
        row = {
            "disease_id": disease_id,
            "label": consensus_row["label"],
        }

        for category in categories:
            row[category] = best.get(disease_id, {}).get(category)

        rows.append(row)

    return {
        "categories": categories,
        "rows": rows,
    }


def build_comparison(
    by_method: dict[str, list[dict[str, Any]]],
    k: int = 10,
    top_n: int = 12,
) -> dict[str, Any]:
    """Build the complete method-comparison payload for one case."""
    norm = normalize_by_method(by_method, k)
    consensus = fuse_rrf(norm, top_n)

    return {
        "k": k,
        "methods": list(norm),
        "short_names": {
            method: short_name(method)
            for method in norm
        },
        "categories": {
            method: method_category(method)
            for method in norm
        },
        "by_method": norm,
        "consensus": consensus,
        "agreement": agreement_jaccard(norm),
        "top_candidate": consensus[0] if consensus else None,
    }
