"""
Shared utilities for RareSim batch evaluation runners.

Imported by run_cpu.py, run_transformer.py, run_llm.py, run_hpo2vec.py.
Do not run this file directly.
"""

import json
from pathlib import Path
from raresim.types.schemas import PatientProfile
from raresim.utils.io import load_json
from raresim.utils.math import get_ancestors_inclusive
from raresim.utils.paths import OUTPUTS_DIR

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# ── Method lists ──────────────────────────────────────────────────────────────

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


# ── Data loading ───────────────────────────────────────────────────────────────


def load_test_cases(path: Path) -> list[tuple[list[str], list[str]]]:
    """Load test cases from JSON. Returns (hpo_terms, ground_truth) tuples."""
    raw = load_json(path)
    return [(entry[0], entry[1]) for entry in raw]


# ── Patient builder ────────────────────────────────────────────────────────────


def build_patient(
    index: int,
    hpo_terms: list[str],
    ancestor_sets: dict,
) -> PatientProfile:
    """Build a PatientProfile with propagated HPO terms."""
    raw_terms = set(hpo_terms)
    propagated = set()
    for term in raw_terms:
        propagated |= get_ancestors_inclusive(term, ancestor_sets)
    return PatientProfile(
        patient_id=f"eval_case_{index:04d}",
        raw_text="",
        hpo_terms=raw_terms,
        propagated_hpo_terms=propagated,
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────


def cache_path_for(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"case_{index:04d}.json"


def load_cache(path: Path) -> dict:
    """Load an existing cache file, or return an empty skeleton."""
    if path.exists():
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"results": {}, "method_elapsed_seconds": {}}


def save_cache(
    path: Path,
    index: int,
    hpo_terms: list[str],
    ground_truth: list[str],
    results: dict,
    method_elapsed: dict[str, float],
    total_elapsed: float,
) -> None:
    """
    Write (or update) a case cache file, merging with any existing results.

    results          — {method_name: [result_dicts]}
    method_elapsed   — {method_name: seconds} for methods run this session
    total_elapsed    — wall-clock seconds for the full case this session
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache(path)

    merged_results = {**existing.get("results", {}), **results}
    merged_elapsed = {**existing.get("method_elapsed_seconds", {}), **method_elapsed}

    # Accumulate total_elapsed across all sessions rather than overwriting
    accumulated_total = round(
        existing.get("total_elapsed_seconds", 0.0) + total_elapsed, 3
    )

    data = {
        "case_index": index,
        "hpo_terms": sorted(hpo_terms),
        "ground_truth": ground_truth,
        "total_elapsed_seconds": accumulated_total,
        "method_elapsed_seconds": {k: round(v, 3) for k, v in merged_elapsed.items()},
        "methods_run": sorted(merged_results.keys()),
        "results": merged_results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def methods_already_cached(cache_file: Path, required_methods: list[str]) -> bool:
    """Return True if every method in required_methods is present in the cache."""
    if not cache_file.exists():
        return False
    existing = load_cache(cache_file)
    cached = set(existing.get("methods_run", []))
    return all(m in cached for m in required_methods)


def serialize_results(results: dict) -> dict[str, list[dict]]:
    """Normalise result objects to plain dicts for JSON serialisation."""
    serialized = {}
    for method, rows in results.items():
        if hasattr(rows, "rankings"):
            serialized[method] = [r.to_dict() for r in rows.rankings]
        elif isinstance(rows, list):
            serialized[method] = [
                r.to_dict() if hasattr(r, "to_dict") else r for r in rows
            ]
        else:
            serialized[method] = []
    return serialized


# ── Console helpers ────────────────────────────────────────────────────────────


def print_header(
    pipeline: str, test_set_path: Path, cache_dir: Path, resume: bool, limit
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RareSim Batch Runner — {pipeline}")
    print(f"{sep}")
    print(f"  Test set : {test_set_path.name}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Resume   : {resume}")
    print(f"  Limit    : {limit if limit else 'all cases'}")
    print(f"{sep}\n")


def print_case(index: int, total: int, hpo_terms, ground_truth) -> None:
    print(
        f"[{index + 1:>4}/{total}] case_{index:04d} | "
        f"{len(hpo_terms)} HPO terms | gt={ground_truth}"
    )


def print_case_ok(
    elapsed: float, total_time: float, processed: int, remaining: int
) -> None:
    avg = total_time / processed
    eta = remaining * avg / 60
    print(f"           ✓ {elapsed:.1f}s | avg={avg:.1f}s | est. remaining={eta:.1f}min")


def print_case_err(e: Exception) -> None:
    print(f"           ✗ ERROR: {e}")


def print_summary(total, processed, skipped, failed, total_time, cache_dir) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Batch complete")
    print(f"{sep}")
    print(f"  Total    : {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped  : {skipped}  (already cached)")
    print(f"  Failed   : {failed}")
    if processed > 0:
        print(f"  Time     : {total_time / 60:.1f} min")
        print(f"  Avg/case : {total_time / processed:.1f} s")
    print(f"  Cache    : {cache_dir}")
    print(f"{sep}\n")


# ── Common CLI args ────────────────────────────────────────────────────────────


def add_common_args(parser) -> None:
    """Add --test-set, --no-resume, --limit, --top-k to an ArgumentParser."""
    parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to test set JSON  (e.g. test_data/test_cases/MME.json)",
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
        help="Only process the first N cases (useful for testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results per method (default: 10)",
    )
