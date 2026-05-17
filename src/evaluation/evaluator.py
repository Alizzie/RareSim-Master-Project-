"""
RareSim Evaluation Script

Loads cached results from run_test_files.py and computes rank-based metrics
for all methods across all pipelines (semantic/set/tfidf/transformer/llm).
Produces a unified comparison table.

Metrics:
    Recall@1  — fraction of cases where correct disease was ranked 1st
    Recall@5  — fraction of cases where correct disease was in top 5
    Recall@10 — fraction of cases where correct disease was in top 10
    MRR       — Mean Reciprocal Rank (1/rank averaged across all cases)
    NDCG@10   — Normalized Discounted Cumulative Gain at cutoff 10
                Uses best rank when multiple ground truth IDs exist.

    Per-case rank matrix shows which method found the correct disease at what rank.

Usage:
    # Evaluate CPU methods only (semantic/set/tfidf)
    python src/evaluation/evaluator.py --cache-dir outputs/evaluation/cache/HMS

    # Evaluate all pipelines
    python src/evaluation/evaluator.py \
  --cache-dir outputs/evaluation/cache/HMS \
  --transformer-cache outputs/evaluation/cache/HMS_transformer \
        --llm-cache outputs/evaluation/cache/HMS_llm

Output:
    outputs/evaluation/HMS_evaluation.json
    outputs/evaluation/HMS_evaluation_summary.txt
    test_data/results/evaluation/HMS_evaluation_summary.txt  ← to tracked by git
""" 

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.io import load_json
from shared.paths import ALIAS_TO_CANONICAL_PATH, OUTPUTS_DIR

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"


# ── ID normalization ───────────────────────────────────────────────────────────


def load_alias_map() -> dict[str, str]:
    if ALIAS_TO_CANONICAL_PATH.exists():
        return load_json(ALIAS_TO_CANONICAL_PATH)
    print("[warning] alias_to_canonical.json not found — using direct ID matching only")
    return {}


def build_reverse_map(alias_map: dict[str, str]) -> dict[str, set[str]]:
    reverse = defaultdict(set)
    for alias, canonical in alias_map.items():
        reverse[canonical].add(alias)
        reverse[canonical].add(canonical)
        reverse[alias].add(alias)
        reverse[alias].add(canonical)
    return reverse


def get_all_equivalent_ids(
    disease_id: str,
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
) -> set[str]:
    ids = {disease_id}
    canonical = alias_map.get(disease_id, disease_id)
    ids.add(canonical)
    ids.update(reverse_map.get(disease_id, set()))
    ids.update(reverse_map.get(canonical, set()))
    return ids


# ── Result loading ─────────────────────────────────────────────────────────────


def get_disease_id_from_result(result: dict) -> str | None:
    """
    Extract disease ID from a result dict regardless of schema.

    Handles three schemas:
    - semantic/set/tfidf: uses 'disease_id'
    - transformer:        uses 'canonical_disease_id'
    - llm:                uses 'ordo_id'
    """
    return (
        result.get("disease_id")
        or result.get("canonical_disease_id")
        or result.get("ordo_id")
    )


def find_rank(
    ground_truth_ids: list[str],
    results: list[dict],
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
) -> int | None:
    """
    Find the best rank of any ground truth disease in the results.

    Returns the rank (1-indexed) or None if not found.
    """
    # Build full set of equivalent IDs for all ground truth diseases
    gt_equivalent = set()
    for gt_id in ground_truth_ids:
        gt_equivalent.update(get_all_equivalent_ids(gt_id, alias_map, reverse_map))

    for result in results:
        result_id = get_disease_id_from_result(result)
        if not result_id:
            continue

        result_equivalent = get_all_equivalent_ids(result_id, alias_map, reverse_map)
        if gt_equivalent & result_equivalent:
            return result.get("rank")

    return None


def load_cache_dir(cache_dir: Path) -> list[dict]:
    """Load all cached case files from a directory."""
    if not cache_dir.exists():
        return []
    files = sorted(cache_dir.glob("case_*.json"))
    cases = []
    for f in files:
        try:
            with f.open() as fp:
                cases.append(json.load(fp))
        except Exception as e:
            print(f"[warning] Could not load {f.name}: {e}")
    return cases


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_ndcg(rank: int | None, top_k: int = 10) -> float:
    """
    Compute NDCG for a single case.

    NDCG@k = 1/log2(rank+1) if rank <= k, else 0.
    Ideal DCG = 1/log2(2) = 1.0 (correct disease at rank 1).
    """
    if rank is None or rank > top_k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def compute_metrics(
    ranks: list[int | None],
    top_k: int = 10,
) -> dict:
    """Compute Recall@1/5/10, MRR, and NDCG@10 from a list of ranks."""
    n = len(ranks)
    if n == 0:
        return {"recall_1": 0, "recall_5": 0, "recall_10": 0, "mrr": 0, "ndcg": 0, "found": 0}

    recall_1 = sum(1 for r in ranks if r == 1) / n
    recall_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    recall_10 = sum(1 for r in ranks if r is not None and r <= 10) / n
    mrr = sum(1 / r for r in ranks if r is not None) / n
    ndcg = sum(compute_ndcg(r, top_k) for r in ranks) / n
    found = sum(1 for r in ranks if r is not None)

    return {
        "recall_1": round(recall_1, 4),
        "recall_5": round(recall_5, 4),
        "recall_10": round(recall_10, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "found": found,
    }


# ── Case merging ──────────────────────────────────────────────────────────────


def merge_cases(
    cpu_cases: list[dict],
    transformer_cases: list[dict],
    llm_cases: list[dict],
) -> list[dict]:
    """
    Merge cases from all three cache directories by case index.

    Each merged case contains results from all available pipelines.
    """
    # Index by case_index
    cpu_by_index = {c["case_index"]: c for c in cpu_cases}
    transformer_by_index = {c["case_index"]: c for c in transformer_cases}
    llm_by_index = {c["case_index"]: c for c in llm_cases}

    all_indices = set(cpu_by_index) | set(transformer_by_index) | set(llm_by_index)
    merged = []

    for index in sorted(all_indices):
        cpu_case = cpu_by_index.get(index, {})
        transformer_case = transformer_by_index.get(index, {})
        llm_case = llm_by_index.get(index, {})

        # Use ground truth from whichever cache has it
        ground_truth = (
            cpu_case.get("ground_truth")
            or transformer_case.get("ground_truth")
            or llm_case.get("ground_truth")
            or []
        )
        hpo_terms = (
            cpu_case.get("hpo_terms")
            or transformer_case.get("hpo_terms")
            or llm_case.get("hpo_terms")
            or []
        )

        # Merge all results
        results = {}
        results.update(cpu_case.get("results", {}))
        results.update(transformer_case.get("results", {}))
        results.update(llm_case.get("results", {}))

        merged.append({
            "case_index": index,
            "hpo_terms": hpo_terms,
            "ground_truth": ground_truth,
            "results": results,
        })

    return merged


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    cases: list[dict],
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
    top_k: int = 10,
) -> dict:
    """
    Evaluate all methods across all cases.

    Returns per-method metrics and per-case rank matrix.
    """
    # Collect all method names
    all_methods = set()
    for case in cases:
        all_methods.update(case.get("results", {}).keys())
    all_methods = sorted(all_methods)

    # Per-method rank lists
    method_ranks: dict[str, list[int | None]] = {m: [] for m in all_methods}

    # Per-case rank matrix
    rank_matrix = []

    for case in cases:
        ground_truth = case.get("ground_truth", [])
        results = case.get("results", {})

        case_ranks = {}
        for method in all_methods:
            method_results = results.get(method, [])
            rank = find_rank(ground_truth, method_results, alias_map, reverse_map)
            method_ranks[method].append(rank)
            case_ranks[method] = rank

        rank_matrix.append({
            "case_index": case["case_index"],
            "ground_truth": ground_truth,
            "ranks": case_ranks,
        })

    # Compute metrics per method
    method_metrics = {}
    for method in all_methods:
        ranks = method_ranks[method]
        method_metrics[method] = compute_metrics(ranks, top_k)

    return {
        "n_cases": len(cases),
        "n_methods": len(all_methods),
        "methods": all_methods,
        "method_metrics": method_metrics,
        "rank_matrix": rank_matrix,
    }


# ── Formatting ────────────────────────────────────────────────────────────────


def format_summary(results: dict, test_set_name: str) -> str:
    n = results["n_cases"]
    metrics = results["method_metrics"]
    rank_matrix = results["rank_matrix"]

    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Evaluation Summary — {test_set_name} ({n} cases)")
    lines.append(f"{'=' * 80}")

    # Method comparison table
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Method Comparison — Recall@k, MRR, NDCG@10")
    lines.append(f"{'=' * 80}")
    lines.append(
        f"  {'Method':<45} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}  Found"
    )
    lines.append(f"  {'-' * 78}")

    # Sort by Recall@10 descending
    sorted_methods = sorted(
        metrics.items(),
        key=lambda x: (-x[1]["recall_10"], -x[1]["mrr"]),
    )

    for method, m in sorted_methods:
        found_str = f"{m['found']}/{n}"
        lines.append(
            f"  {method:<45} "
            f"{m['recall_1']:>6.4f} "
            f"{m['recall_5']:>6.4f} "
            f"{m['recall_10']:>6.4f} "
            f"{m['mrr']:>6.4f} "
            f"{m['ndcg']:>6.4f}  "
            f"{found_str}"
        )
    lines.append(f"{'=' * 80}")

    # Per-case rank matrix
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Per-Case Rank Matrix  (- = not found in top 10)")
    lines.append(f"{'=' * 80}")

    all_methods = results["methods"]
    method_headers = "  ".join(f"{m[:12]:>12}" for m in all_methods)
    lines.append(f"  {'Case':<12} {'GT':<25}  {method_headers}")
    lines.append(f"  {'-' * 78}")

    for row in rank_matrix:
        case_id = f"case_{row['case_index']:04d}"
        gt_str = ",".join(row["ground_truth"])[:24]
        rank_strs = "  ".join(
            f"{str(row['ranks'].get(m) or '-'):>12}" for m in all_methods
        )
        lines.append(f"  {case_id:<12} {gt_str:<25}  {rank_strs}")

    lines.append(f"{'=' * 80}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    test_set_name = args.cache_dir.name

    print(f"\nLoading cached results...")
    cpu_cases = load_cache_dir(args.cache_dir)
    transformer_cases = load_cache_dir(args.transformer_cache) if args.transformer_cache else []
    llm_cases = load_cache_dir(args.llm_cache) if args.llm_cache else []

    print(f"  CPU cases        : {len(cpu_cases)}")
    print(f"  Transformer cases: {len(transformer_cases)}")
    print(f"  LLM cases        : {len(llm_cases)}")

    print("\nMerging cases...")
    cases = merge_cases(cpu_cases, transformer_cases, llm_cases)
    print(f"  Total merged cases: {len(cases)}")

    print("\nLoading alias map...")
    alias_map = load_alias_map()
    reverse_map = build_reverse_map(alias_map)

    print("\nEvaluating...")
    results = evaluate(cases, alias_map, reverse_map, top_k=args.top_k)

    # Print summary
    summary = format_summary(results, test_set_name)
    print(summary)

    # Save outputs
    # Save outputs — also save to results/ for version control
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = PROJECT_ROOT / "test_data" / "results" / "evaluation"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVALUATION_DIR / f"{test_set_name}_evaluation.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON : {json_path}")

    txt_path = EVALUATION_DIR / f"{test_set_name}_evaluation_summary.txt"
    txt_path.write_text(summary)
    print(f"Saved TXT  : {txt_path}")

    # Copy to results/ for git tracking
    results_txt_path = RESULTS_DIR / f"{test_set_name}_evaluation_summary.txt"
    results_txt_path.write_text(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cache directory for CPU results (semantic/set/tfidf)",
    )
    parser.add_argument(
        "--transformer-cache",
        type=Path,
        default=None,
        help="Cache directory for transformer results (optional)",
    )
    parser.add_argument(
        "--llm-cache",
        type=Path,
        default=None,
        help="Cache directory for LLM results (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff for Recall@k and NDCG (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
