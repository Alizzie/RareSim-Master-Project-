"""Exploratory RRF experiment combining Rarefully with external systems.

External source selection:
    HMS                -> PhenoBrain API
    MME                -> PhenoBrain API
    PUMCH_L            -> PhenoBrain API
    PUMCH-ADM          -> PhenoBrain API
    RAMEDIS            -> PhenoBrain API

    LIRICAL            -> LIRICAL

"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from scripts.evaluation._batch_utils import EVALUATION_DIR
    from scripts.evaluation.evaluator import (
        RRF_K,
        build_reverse_map,
        compute_metrics,
        compute_ndcg_for_case,
        count_distinct_ground_truth,
        evaluate,
        find_all_matched_ranks,
        find_rank,
        get_disease_id_from_result,
        load_alias_map,
        load_cache_dir,
    )
except Exception:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.evaluation._batch_utils import EVALUATION_DIR
    from scripts.evaluation.evaluator import (
        RRF_K,
        build_reverse_map,
        compute_metrics,
        compute_ndcg_for_case,
        count_distinct_ground_truth,
        evaluate,
        find_all_matched_ranks,
        find_rank,
        get_disease_id_from_result,
        load_alias_map,
        load_cache_dir,
    )

from raresim.utils.paths import OUTPUTS_DIR


EXPERIMENT_DIR = OUTPUTS_DIR / "ensemble_external_experiment"

LIRICAL_DIR = (
    OUTPUTS_DIR
    / "validation_tools"
    / "lirical_benchmarks"
    / "cache"
)


EXTERNAL_SYSTEM = {
    "HMS": "phenobrain",
    "MME": "phenobrain",
    "PUMCH_L": "phenobrain",
    "PUMCH-ADM": "phenobrain",
    "RAMEDIS": "phenobrain",

    "LIRICAL": "lirical",

}


def canonical_id(
    disease_ids: list[str],
    alias_map: dict[str, str],
) -> str | None:
    """Choose one canonical disease ID for fusion."""

    if not disease_ids:
        return None

    canonical = [
        alias_map.get(disease_id, disease_id)
        for disease_id in disease_ids
        if disease_id
    ]

    # Prefer an ORPHA identifier when one is available
    for disease_id in canonical:
        if disease_id.startswith("ORPHA:"):
            return disease_id

    # Then OMIM
    for disease_id in canonical:
        if disease_id.startswith("OMIM:"):
            return disease_id

    return canonical[0] if canonical else None


def load_json_external(
    system: str,
    dataset: str,
    case_index: int,
    alias_map: dict[str, str],
    depth: int,
) -> list[dict]:

    path = (
        EXPERIMENT_DIR
        / system
        / dataset
        / f"case_{case_index:04d}.json"
    )

    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    ranking = []

    for row in data.get("results", []):

        rank = row.get("rank")

        if rank is None or rank > depth:
            continue

        if system == "phenobrain":

            disease_id = canonical_id(
                row.get("disease_ids", []),
                alias_map,
            )

        else:

            raw_id = row.get("disease_id")

            disease_id = canonical_id(
                [raw_id] if raw_id else [],
                alias_map,
            )

        if disease_id:
            ranking.append(
                {
                    "disease_id": disease_id,
                    "rank": rank,
                }
            )

    return ranking


def find_lirical_dataset_dir(dataset: str) -> Path | None:
    """Resolve case-insensitive LIRICAL dataset directory."""

    if not LIRICAL_DIR.exists():
        return None

    target = dataset.lower()

    for path in LIRICAL_DIR.iterdir():
        if path.is_dir() and path.name.lower() == target:
            return path

    return None


def load_lirical(
    dataset: str,
    case_index: int,
    alias_map: dict[str, str],
    depth: int,
) -> list[dict]:

    dataset_dir = find_lirical_dataset_dir(dataset)

    if dataset_dir is None:
        return []

    # the original runner uses <dataset>_case_XXXX.tsv.
    candidates = list(
        dataset_dir.glob(f"*_case_{case_index:04d}.tsv")
    )

    if not candidates:
        return []

    path = candidates[0]

    lines = [
        line
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.startswith("!")
    ]

    if not lines:
        return []

    reader = csv.DictReader(
        lines,
        delimiter="\t",
    )

    ranking = []

    for row in reader:

        try:
            rank = int(
                str(
                    row.get("rank", "0")
                ).replace(",", "")
            )
        except ValueError:
            continue

        if rank > depth:
            continue

        disease_id = (
            row.get("diseaseCurie")
            or row.get("diseaseId")
            or row.get("disease_id")
            or ""
        ).strip()

        disease_id = canonical_id(
            [disease_id],
            alias_map,
        )

        if disease_id:
            ranking.append(
                {
                    "disease_id": disease_id,
                    "rank": rank,
                }
            )

    return ranking


def load_external_ranking(
    dataset: str,
    case_index: int,
    alias_map: dict[str, str],
    depth: int,
) -> list[dict]:

    system = EXTERNAL_SYSTEM[dataset]

    if system == "lirical":
        return load_lirical(
            dataset,
            case_index,
            alias_map,
            depth,
        )

    return load_json_external(
        system,
        dataset,
        case_index,
        alias_map,
        depth,
    )


def canonicalize_ranking(
    results: list[dict],
    alias_map: dict[str, str],
    depth: int,
) -> list[dict]:
    """
    Canonicalize one ranked list before RRF.

    If multiple identifiers resolve to the same canonical disease,
    retain only its best rank within this ranking.
    """
    best_rank: dict[str, int] = {}

    for result in results:
        rank = result.get("rank")

        if rank is None or rank > depth:
            continue

        disease_id = get_disease_id_from_result(result)

        if not disease_id:
            continue

        canonical = alias_map.get(disease_id, disease_id)

        if (
            canonical not in best_rank
            or rank < best_rank[canonical]
        ):
            best_rank[canonical] = rank

    return [
        {
            "disease_id": disease_id,
            "rank": rank,
        }
        for disease_id, rank in sorted(
            best_rank.items(),
            key=lambda item: item[1],
        )
    ]


def compute_rrf(
    case_results: dict[str, list[dict]],
    methods: list[str],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """Compute equal-weight Reciprocal Rank Fusion for one case."""

    scores: dict[str, float] = defaultdict(float)

    for method in methods:
        for result in case_results.get(method, []):
            disease_id = get_disease_id_from_result(result)
            rank = result.get("rank")

            if not disease_id or rank is None:
                continue

            scores[disease_id] += 1.0 / (k + rank)

    ordered = sorted(
        scores.items(),
        key=lambda item: -item[1],
    )[:top_k]

    return [
        {
            "disease_id": disease_id,
            "rank": rank,
        }
        for rank, (disease_id, _) in enumerate(
            ordered,
            start=1,
        )
    ]


def evaluate_dataset(
    dataset: str,
    fusion_depth: int,
    top_k: int,
):
    """Evaluate two exploratory external-augmented RRF variants.

    A: all Rarefully base methods + the selected external system.
    B: best individual Rarefully method + the selected external system.

    All rankings are canonicalized before fusion. Existing caches are read-only.
    """

    print(f"\n{'=' * 80}")
    print(dataset)
    print(f"{'=' * 80}")

    cache_dir = EVALUATION_DIR / dataset / "cache"
    cases = load_cache_dir(cache_dir)

    if not cases:
        print("No Rarefully cache found.")
        return None

    alias_map = load_alias_map()
    reverse_map = build_reverse_map(alias_map)

    rarefully_eval = evaluate(
        cases,
        alias_map,
        reverse_map,
        top_k,
    )

    base_methods = sorted(
        {
            method
            for case in cases
            for method in case.get("results", {}).keys()
        }
    )

    # Select the strongest individual Rarefully method on this dataset
    # Variant B is therefore exploratory / retrospective
    base_metrics = {
        method: rarefully_eval["method_metrics"][method]
        for method in base_methods
    }

    best_method = max(
        base_metrics,
        key=lambda method: (
            base_metrics[method]["recall_10"],
            base_metrics[method]["mrr"],
        ),
    )
    best_metrics = base_metrics[best_method]

    rarefully_rrf = rarefully_eval["method_metrics"]["ensemble_rrf"]

    external_ranks: list[int | None] = []
    external_ndcg: list[float] = []

    # A: all Rarefully methods + external.
    all_plus_external_ranks: list[int | None] = []
    all_plus_external_ndcg: list[float] = []

    # B: best Rarefully method + external.
    best_plus_external_ranks: list[int | None] = []
    best_plus_external_ndcg: list[float] = []

    missing_external = []

    for case in cases:
        case_index = case["case_index"]
        ground_truth = case.get("ground_truth", [])

        external = load_external_ranking(
            dataset,
            case_index,
            alias_map,
            fusion_depth,
        )

        if not external:
            missing_external.append(case_index)
            continue

        # Canonicalize/deduplicate the external ranking before both evaluation
        # and fusion, so aliases of the same disease cannot contribute twice
        external = canonicalize_ranking(
            external,
            alias_map,
            fusion_depth,
        )

        n_relevant = count_distinct_ground_truth(
            ground_truth,
            alias_map,
            reverse_map,
        )

        # External system alone
        ext_rank = find_rank(
            ground_truth,
            external,
            alias_map,
            reverse_map,
        )
        ext_matched = find_all_matched_ranks(
            ground_truth,
            external,
            alias_map,
            reverse_map,
        )
        external_ranks.append(ext_rank)
        external_ndcg.append(
            compute_ndcg_for_case(
                ext_matched,
                n_relevant,
                top_k,
            )
        )

        # Canonicalize every Rarefully ranking before fusion
        rarefully_inputs: dict[str, list[dict]] = {}
        for method in base_methods:
            rarefully_inputs[method] = canonicalize_ranking(
                case.get("results", {}).get(method, []),
                alias_map,
                fusion_depth,
            )

        # Variant A: all Rarefully base methods + external system
        all_inputs = dict(rarefully_inputs)
        all_inputs["external_system"] = external

        all_augmented = compute_rrf(
            all_inputs,
            base_methods + ["external_system"],
            top_k,
        )

        all_rank = find_rank(
            ground_truth,
            all_augmented,
            alias_map,
            reverse_map,
        )
        all_matched = find_all_matched_ranks(
            ground_truth,
            all_augmented,
            alias_map,
            reverse_map,
        )
        all_plus_external_ranks.append(all_rank)
        all_plus_external_ndcg.append(
            compute_ndcg_for_case(
                all_matched,
                n_relevant,
                top_k,
            )
        )

        # Variant B: strongest individual Rarefully method + external system
        best_inputs = {
            best_method: rarefully_inputs[best_method],
            "external_system": external,
        }

        best_augmented = compute_rrf(
            best_inputs,
            [best_method, "external_system"],
            top_k,
        )

        best_rank = find_rank(
            ground_truth,
            best_augmented,
            alias_map,
            reverse_map,
        )
        best_matched = find_all_matched_ranks(
            ground_truth,
            best_augmented,
            alias_map,
            reverse_map,
        )
        best_plus_external_ranks.append(best_rank)
        best_plus_external_ndcg.append(
            compute_ndcg_for_case(
                best_matched,
                n_relevant,
                top_k,
            )
        )

    if missing_external:
        print(
            f"Missing external rankings: "
            f"{len(missing_external)}/{len(cases)}"
        )
        print(
            "Experiment skipped until external "
            "coverage is complete."
        )
        return None

    external_metrics = compute_metrics(
        external_ranks,
        external_ndcg,
        top_k,
    )

    all_plus_external_metrics = compute_metrics(
        all_plus_external_ranks,
        all_plus_external_ndcg,
        top_k,
    )

    best_plus_external_metrics = compute_metrics(
        best_plus_external_ranks,
        best_plus_external_ndcg,
        top_k,
    )

    result = {
        "dataset": dataset,
        "external_system": EXTERNAL_SYSTEM[dataset],
        "n_cases": len(cases),
        "fusion_depth": fusion_depth,
        "best_rarefully_method": best_method,
        "best_rarefully": best_metrics,
        "rarefully_rrf": rarefully_rrf,
        "external": external_metrics,
        "all_rarefully_plus_external_rrf": all_plus_external_metrics,
        "best_rarefully_plus_external_rrf": best_plus_external_metrics,
    }

    print(
        f"Best Rarefully method        : "
        f"{best_method:<30} "
        f"R@10={best_metrics['recall_10']:.4f}"
    )
    print(
        f"Rarefully RRF                : "
        f"R@10={rarefully_rrf['recall_10']:.4f}"
    )
    print(
        f"External ({EXTERNAL_SYSTEM[dataset]})"
        f"          : R@10={external_metrics['recall_10']:.4f}"
    )
    print(
        f"A) All Rarefully + external : "
        f"R@10={all_plus_external_metrics['recall_10']:.4f}"
    )
    print(
        f"B) Best Rarefully + external: "
        f"R@10={best_plus_external_metrics['recall_10']:.4f}"
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(EXTERNAL_SYSTEM.keys()),
    )

    parser.add_argument(
        "--fusion-depth",
        type=int,
        default=10,
        help="Ranking depth contributed by each system.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results = []

    for dataset in args.datasets:

        if dataset not in EXTERNAL_SYSTEM:
            print(
                f"Unknown dataset: {dataset}"
            )
            continue

        result = evaluate_dataset(
            dataset,
            args.fusion_depth,
            args.top_k,
        )

        if result is not None:
            results.append(result)

    output = (
        EXPERIMENT_DIR
        / "external_rrf_results.json"
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load results from previous runs, if available
    existing_results = []

    if output.exists():
        with output.open("r", encoding="utf-8") as f:
            existing_results = json.load(f)

    # Store results by dataset so rerunning one dataset
    # replaces only that dataset's previous result
    merged = {
        result["dataset"]: result
        for result in existing_results
    }

    for result in results:
        merged[result["dataset"]] = result

    merged_results = [
        merged[dataset]
        for dataset in sorted(merged)
    ]

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            merged_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
