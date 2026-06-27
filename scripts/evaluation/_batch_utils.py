"""
Shared utilities for RareSim batch evaluation runners.

Imported by the batch runners in ``scripts/evaluation``.
"""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import get_ancestors_inclusive
from raresim.utils.io import load_json
from raresim.utils.paths import OUTPUTS_DIR

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

SEMANTIC_METHODS = [
    "semantic_resnik_bma",
    "semantic_lin_bma",
    "semantic_jiang_conrath_bma",
]

SET_BASED_METHODS = [
    "set_cosine",
    "set_jaccard",
    "set_dice",
    "set_overlap",
]

TFIDF_METHODS = ["tfidf"]
CPU_METHODS = SEMANTIC_METHODS + SET_BASED_METHODS + TFIDF_METHODS


def load_test_cases(path: Path) -> list[tuple[list[str], list[str]]]:
    """Load test cases from JSON as ``(hpo_terms, ground_truth)`` tuples."""
    raw = load_json(path)
    return [(entry[0], entry[1]) for entry in raw]


def build_patient(
    index: int,
    hpo_terms: list[str],
    ancestor_sets: dict[str, set[str]],
) -> PatientProfile:
    """Build a ``PatientProfile`` with propagated HPO terms."""
    raw_terms = set(hpo_terms)
    propagated: set[str] = set()

    for term in raw_terms:
        propagated |= get_ancestors_inclusive(term, ancestor_sets)

    return PatientProfile(
        patient_id=f"eval_case_{index:04d}",
        raw_text="",
        hpo_terms=raw_terms,
        propagated_hpo_terms=propagated,
    )


def cache_path_for(cache_dir: Path, index: int) -> Path:
    """Return the cache path for one evaluation case."""
    return cache_dir / f"case_{index:04d}.json"


def load_cache(path: Path) -> dict[str, Any]:
    """Load an existing cache file, or return an empty cache skeleton."""
    if path.exists():
        try:
            with path.open(encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except (JSONDecodeError, OSError) as error:
            print(f"[warning] Could not load cache {path}: {error}")

    return {"results": {}, "method_elapsed_seconds": {}}


def save_cache(  # pylint: disable=too-many-positional-arguments, too-many-arguments
    path: Path,
    index: int,
    hpo_terms: list[str],
    ground_truth: list[str],
    results: dict[str, list[dict[str, Any]]],
    method_elapsed: dict[str, float],
    total_elapsed: float,
) -> None:
    """Write or update one case cache file, merging with existing results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache(path)

    merged_results = {**existing.get("results", {}), **results}
    merged_elapsed = {
        **existing.get("method_elapsed_seconds", {}),
        **method_elapsed,
    }
    accumulated_total = round(
        existing.get("total_elapsed_seconds", 0.0) + total_elapsed,
        3,
    )

    data = {
        "case_index": index,
        "hpo_terms": sorted(hpo_terms),
        "ground_truth": ground_truth,
        "total_elapsed_seconds": accumulated_total,
        "method_elapsed_seconds": {
            method: round(seconds, 3)
            for method, seconds in merged_elapsed.items()
        },
        "methods_run": sorted(merged_results.keys()),
        "results": merged_results,
    }

    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def methods_already_cached(cache_file: Path, required_methods: list[str]) -> bool:
    """Return True if every required method is already present in the cache."""
    if not cache_file.exists():
        return False

    existing = load_cache(cache_file)
    cached = set(existing.get("methods_run", []))
    return all(method in cached for method in required_methods)


def serialize_results(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize result objects to plain dictionaries for JSON serialization."""
    serialized: dict[str, list[dict[str, Any]]] = {}

    for method, rows in results.items():
        if hasattr(rows, "rankings"):
            serialized[method] = [row.to_dict() for row in rows.rankings]
        elif isinstance(rows, list):
            serialized[method] = [
                row.to_dict() if hasattr(row, "to_dict") else row
                for row in rows
            ]
        else:
            serialized[method] = []

    return serialized


def print_header(
    pipeline: str,
    test_set_path: Path,
    cache_dir: Path,
    resume: bool,
    limit: int | None,
) -> None:
    """Print a standard batch-runner header."""
    separator = "=" * 64
    print(f"\n{separator}")
    print(f"  RareSim Batch Runner - {pipeline}")
    print(separator)
    print(f"  Test set : {test_set_path.name}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Resume   : {resume}")
    print(f"  Limit    : {limit if limit else 'all cases'}")
    print(f"{separator}\n")


def print_case(index: int, total: int, hpo_terms: list[str], ground_truth: list[str]) -> None:
    """Print one case progress line."""
    print(
        f"[{index + 1:>4}/{total}] case_{index:04d} | "
        f"{len(hpo_terms)} HPO terms | gt={ground_truth}"
    )


def print_case_ok(
    elapsed: float,
    total_time: float,
    processed: int,
    remaining: int,
) -> None:
    """Print a successful case progress line with timing."""
    avg = total_time / processed
    eta = remaining * avg / 60
    print(
        f"           OK {elapsed:.1f}s | avg={avg:.1f}s | "
        f"est. remaining={eta:.1f}min"
    )


def print_case_err(error: Exception) -> None:
    """Print a failed case message."""
    print(f"           ERROR: {error}")


def print_summary(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    total: int,
    processed: int,
    skipped: int,
    failed: int,
    total_time: float,
    cache_dir: Path,
) -> None:
    """Print a standard batch-runner summary."""
    separator = "=" * 64
    print(f"\n{separator}")
    print("  Batch complete")
    print(separator)
    print(f"  Total    : {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped  : {skipped}  (already cached)")
    print(f"  Failed   : {failed}")

    if processed > 0:
        print(f"  Time     : {total_time / 60:.1f} min")
        print(f"  Avg/case : {total_time / processed:.1f} s")

    print(f"  Cache    : {cache_dir}")
    print(f"{separator}\n")


def add_common_args(parser: Any) -> None:
    """Add common batch-runner CLI arguments to an ``ArgumentParser``."""
    parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to test set JSON, e.g. test_data/test_cases/MME.json",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun all cases even if already cached",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N cases, useful for testing",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results per method, default: 10",
    )
