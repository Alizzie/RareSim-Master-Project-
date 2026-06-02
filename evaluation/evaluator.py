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
    Method agreement analysis shows consensus/hard/easy/unique-find breakdowns.

Usage:
    # Evaluate CPU methods only (semantic/set/tfidf)
    python src/evaluation/evaluator.py --cache-dir outputs/evaluation/cache/HMS

    # Evaluate all pipelines
    python src/evaluation/evaluator.py \
  --cache-dir outputs/evaluation/cache/MME \
  --transformer-cache outputs/evaluation/cache/MME_transformer \
        --llm-cache outputs/evaluation/cache/MME_llm \
        --hpo2vec-cache outputs/evaluation/cache/MME_hpo2vec

Output:
    outputs/evaluation/MME_evaluation.json
    outputs/evaluation/MME_evaluation_summary.txt
    test_data/results/evaluation/MME_evaluation_summary.txt  ← to tracked by git
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
    hpo2vec_cases: list[dict] = [],
) -> list[dict]:
    """
    Merge cases from all cache directories by case index.

    Each merged case contains results from all available pipelines.
    """
    cpu_by_index = {c["case_index"]: c for c in cpu_cases}
    transformer_by_index = {c["case_index"]: c for c in transformer_cases}
    llm_by_index = {c["case_index"]: c for c in llm_cases}
    hpo2vec_by_index = {c["case_index"]: c for c in hpo2vec_cases}

    all_indices = (
        set(cpu_by_index)
        | set(transformer_by_index)
        | set(llm_by_index)
        | set(hpo2vec_by_index)
    )
    merged = []

    for index in sorted(all_indices):
        cpu_case = cpu_by_index.get(index, {})
        transformer_case = transformer_by_index.get(index, {})
        llm_case = llm_by_index.get(index, {})
        hpo2vec_case = hpo2vec_by_index.get(index, {})

        ground_truth = (
            cpu_case.get("ground_truth")
            or transformer_case.get("ground_truth")
            or llm_case.get("ground_truth")
            or hpo2vec_case.get("ground_truth")
            or []
        )
        hpo_terms = (
            cpu_case.get("hpo_terms")
            or transformer_case.get("hpo_terms")
            or llm_case.get("hpo_terms")
            or hpo2vec_case.get("hpo_terms")
            or []
        )

        results = {}
        results.update(cpu_case.get("results", {}))
        results.update(transformer_case.get("results", {}))
        results.update(llm_case.get("results", {}))
        results.update(hpo2vec_case.get("results", {}))

        merged.append({
            "case_index": index,
            "hpo_terms": hpo_terms,
            "ground_truth": ground_truth,
            "results": results,
        })

    return merged


# ── RRF ensemble ─────────────────────────────────────────────────────────────

RRF_K = 60
RRF_METHOD_NAME = "ensemble_rrf"
RRF_WEIGHTED_NAME = "ensemble_rrf_weighted"
RRF_TOP_NAME = "ensemble_rrf_top"
RRF_MIN_RECALL10 = 0.10  # minimum R@10 for a method to enter top ensemble


def compute_rrf(
    case_results: dict[str, list[dict]],
    all_methods: list[str],
    top_k: int = 10,
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Compute Reciprocal Rank Fusion across all methods for a single case.

    RRF(d) = sum_{r in R} w_r / (k + r(d))

    If weights is None, all methods are weighted equally (w_r = 1.0).
    If weights is provided, each method is weighted by its R@10 score.
    Diseases are then re-ranked by descending RRF score.
    Returns a list of dicts with 'disease_id' and 'rank' (1-indexed).
    """
    rrf_scores: dict[str, float] = defaultdict(float)

    for method in all_methods:
        w = weights[method] if weights else 1.0
        if w == 0:
            continue
        for result in case_results.get(method, []):
            disease_id = get_disease_id_from_result(result)
            rank = result.get("rank")
            if disease_id and rank is not None:
                rrf_scores[disease_id] += w / (k + rank)

    sorted_diseases = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [
        {"disease_id": disease_id, "rank": new_rank}
        for new_rank, (disease_id, _) in enumerate(sorted_diseases[:top_k], start=1)
    ]


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
    Three RRF ensemble variants are computed to try to see if different configurations improve performance for the ensemble method:
      - ensemble_rrf         : equal weights, all methods
      - ensemble_rrf_weighted: weighted by each method's R@10
      - ensemble_rrf_top     : equal weights, only methods with R@10 >= threshold
    """
    base_methods = set()
    for case in cases:
        base_methods.update(case.get("results", {}).keys())
    base_methods = sorted(base_methods)

    all_methods = base_methods + [RRF_METHOD_NAME, RRF_WEIGHTED_NAME, RRF_TOP_NAME]
    method_ranks: dict[str, list[int | None]] = {m: [] for m in all_methods}

    # ── Pass 1: compute per-method ranks ─────────────────────────────────────
    rank_matrix_pass1 = []
    for case in cases:
        ground_truth = case.get("ground_truth", [])
        results = case.get("results", {})
        case_ranks = {}
        for method in base_methods:
            method_results = results.get(method, [])
            rank = find_rank(ground_truth, method_results, alias_map, reverse_map)
            method_ranks[method].append(rank)
            case_ranks[method] = rank
        rank_matrix_pass1.append((case, case_ranks))

    # ── Compute per-method R@10 for weighting ─────────────────────────────────
    n = len(cases)
    method_recall10 = {
        method: sum(1 for r in method_ranks[method] if r is not None and r <= 10) / n
        for method in base_methods
    }

    # Top methods: only those above the minimum R@10 threshold
    top_methods = [m for m in base_methods if method_recall10[m] >= RRF_MIN_RECALL10]
    if not top_methods:
        top_methods = base_methods  # fallback: use all if none qualify

    # ── Pass 2: compute RRF variants ─────────────────────────────────────────
    rank_matrix = []
    for case, case_ranks in rank_matrix_pass1:
        ground_truth = case.get("ground_truth", [])
        results = case.get("results", {})

        # Variant 1: equal weights, all methods
        rrf_results = compute_rrf(results, base_methods, top_k)
        rrf_rank = find_rank(ground_truth, rrf_results, alias_map, reverse_map)
        method_ranks[RRF_METHOD_NAME].append(rrf_rank)
        case_ranks[RRF_METHOD_NAME] = rrf_rank

        # Variant 2: weighted by R@10, all methods
        rrf_w_results = compute_rrf(results, base_methods, top_k, weights=method_recall10)
        rrf_w_rank = find_rank(ground_truth, rrf_w_results, alias_map, reverse_map)
        method_ranks[RRF_WEIGHTED_NAME].append(rrf_w_rank)
        case_ranks[RRF_WEIGHTED_NAME] = rrf_w_rank

        # Variant 3: equal weights, top methods only (R@10 >= threshold)
        rrf_top_results = compute_rrf(results, top_methods, top_k)
        rrf_top_rank = find_rank(ground_truth, rrf_top_results, alias_map, reverse_map)
        method_ranks[RRF_TOP_NAME].append(rrf_top_rank)
        case_ranks[RRF_TOP_NAME] = rrf_top_rank

        rank_matrix.append({
            "case_index": case["case_index"],
            "ground_truth": ground_truth,
            "ranks": case_ranks,
        })

    method_metrics = {}
    for method in all_methods:
        method_metrics[method] = compute_metrics(method_ranks[method], top_k)

    return {
        "n_cases": len(cases),
        "n_methods": len(all_methods),
        "methods": all_methods,
        "method_metrics": method_metrics,
        "rank_matrix": rank_matrix,
        "rrf_top_methods": top_methods,
        "rrf_method_weights": method_recall10,
    }


# ── Method agreement analysis ─────────────────────────────────────────────────


def compute_agreement(results: dict) -> dict:
    """
    Compute method agreement statistics across all cases.

    Returns consensus/hard/easy/unique counts, per-count histogram,
    per-rank histogram, unique-find details, and hard-case IDs.
    """
    rank_matrix = results["rank_matrix"]
    all_methods = results["methods"]
    n_methods = len(all_methods)

    consensus_count = 0   # all methods found it
    hard_count = 0        # no method found it
    easy_count = 0        # at least one method ranked it #1
    unique_find_count = 0 # exactly one method found it

    # How many methods found the correct disease per case
    found_by_n: dict[int, int] = defaultdict(int)

    # How many methods found correct disease at each rank (rank 1..top_k)
    rank_histogram: dict[int, int] = defaultdict(int)

    unique_finds: list[dict] = []   # cases where exactly one method found it
    hard_cases: list[str] = []      # cases where no method found it

    for row in rank_matrix:
        case_id = f"case_{row['case_index']:04d}"
        gt = row["ground_truth"]
        ranks = row["ranks"]  # method -> rank | None

        found_methods = [m for m in all_methods if ranks.get(m) is not None]
        n_found = len(found_methods)
        found_by_n[n_found] += 1

        # Accumulate rank histogram
        for m in all_methods:
            r = ranks.get(m)
            if r is not None:
                rank_histogram[r] += 1

        if n_found == n_methods:
            consensus_count += 1
        if n_found == 0:
            hard_count += 1
            hard_cases.append(case_id)
        if any(ranks.get(m) == 1 for m in all_methods):
            easy_count += 1
        if n_found == 1:
            unique_find_count += 1
            solo_method = found_methods[0]
            unique_finds.append({
                "case_id": case_id,
                "gt": gt,
                "method": solo_method,
                "rank": ranks[solo_method],
            })

    return {
        "consensus_count": consensus_count,
        "hard_count": hard_count,
        "easy_count": easy_count,
        "unique_find_count": unique_find_count,
        "found_by_n": dict(found_by_n),
        "rank_histogram": dict(rank_histogram),
        "unique_finds": unique_finds,
        "hard_cases": hard_cases,
        "n_methods": n_methods,
    }


# ── Formatting ────────────────────────────────────────────────────────────────

_BAR_FULL = "█"
_BAR_WIDTH = 40  # max bar width in characters


def _bar(value: int, max_value: int) -> str:
    if max_value == 0:
        return ""
    filled = round(_BAR_WIDTH * value / max_value)
    return _BAR_FULL * filled


def format_agreement_section(agreement: dict, n_cases: int) -> str:
    lines = []
    w = 72
    lines.append(f"\n{'=' * w}")
    lines.append(f"  Method Agreement Analysis")
    lines.append(f"{'=' * w}")

    lines.append(f"  Total cases : {n_cases}")
    lines.append(
        f"  Consensus   : {agreement['consensus_count']} cases"
        f" — all methods found it"
    )
    lines.append(
        f"  Hard cases  : {agreement['hard_count']} cases"
        f" — no method found it"
    )
    lines.append(
        f"  Easy cases  : {agreement['easy_count']} cases"
        f" — at least one method ranked it #1"
    )
    lines.append(
        f"  Unique finds: {agreement['unique_find_count']} cases"
        f" — only one method found it"
    )

    # Per-case histogram: how many methods found it
    lines.append(f"\n  How many methods found the correct disease per case:")
    found_by_n = agreement["found_by_n"]
    n_methods = agreement["n_methods"]
    max_val = max(found_by_n.values(), default=1)
    for k in range(n_methods + 1):
        count = found_by_n.get(k, 0)
        bar = _bar(count, max_val)
        lines.append(f"    {k} methods: {count:>3} cases  {bar}")

    # Rank histogram
    lines.append(f"\n  How many methods found correct disease at each rank:")
    rank_hist = agreement["rank_histogram"]
    if rank_hist:
        max_rank = max(rank_hist.keys())
        max_hist_val = max(rank_hist.values(), default=1)
        for r in range(1, max_rank + 1):
            count = rank_hist.get(r, 0)
            bar = _bar(count, max_hist_val)
            lines.append(f"    rank_{r}: {count:>2}  {bar}")

    # Unique finds
    lines.append(f"\n  Unique finds (only one method found it):")
    if agreement["unique_finds"]:
        for uf in agreement["unique_finds"]:
            gt_str = str(uf["gt"])
            lines.append(
                f"    {uf['case_id']} | gt={gt_str} | {uf['method']} @ rank {uf['rank']}"
            )
    else:
        lines.append("    (none)")

    # Hard cases
    lines.append(f"\n  Hard cases (no method found correct disease):")
    if agreement["hard_cases"]:
        for hc in agreement["hard_cases"]:
            lines.append(f"    {hc}")
    else:
        lines.append("    (none)")

    lines.append(f"{'=' * w}")
    return "\n".join(lines)


def format_summary(results: dict, test_set_name: str) -> str:
    n = results["n_cases"]
    metrics = results["method_metrics"]
    rank_matrix = results["rank_matrix"]

    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Evaluation Summary — {test_set_name} ({n} cases)")
    lines.append(f"{'=' * 80}")

    # ── Method comparison table ──────────────────────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Method Comparison — Recall@k, MRR, NDCG@10")
    lines.append(f"{'=' * 80}")
    lines.append(
        f"  {'Method':<45} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}  Found"
    )
    lines.append(f"  {'-' * 78}")

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

    # ── RRF Ensemble Configuration ───────────────────────────────────────────
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  RRF Ensemble Configuration")
    lines.append(f"{'=' * 80}")
    top_methods = results.get("rrf_top_methods", [])
    weights = results.get("rrf_method_weights", {})
    lines.append(f"  Threshold  : R@10 >= {RRF_MIN_RECALL10} for ensemble_rrf_top")
    lines.append(f"  Top methods: {len(top_methods)} / {results['n_methods'] - 3} base methods")
    for m in top_methods:
        lines.append(f"    {m:<45} R@10={weights.get(m, 0):.4f}")
    lines.append(f"{'=' * 80}")

    # ── Method agreement analysis ────────────────────────────────────────────
    agreement = compute_agreement(results)
    lines.append(format_agreement_section(agreement, n))

    # ── Per-case rank matrix ─────────────────────────────────────────────────
    all_methods = results["methods"]
    lines.append(f"\n{'=' * 80}")
    lines.append(f"  Per-Case Rank Matrix  (- = not found in top 10)")
    lines.append(f"{'=' * 80}")

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
    hpo2vec_cases = load_cache_dir(args.hpo2vec_cache) if args.hpo2vec_cache else []

    print(f"  CPU cases        : {len(cpu_cases)}")
    print(f"  Transformer cases: {len(transformer_cases)}")
    print(f"  LLM cases        : {len(llm_cases)}")
    print(f"  HPO2Vec cases    : {len(hpo2vec_cases)}")

    print("\nMerging cases...")
    cases = merge_cases(cpu_cases, transformer_cases, llm_cases, hpo2vec_cases)
    print(f"  Total merged cases: {len(cases)}")

    print("\nLoading alias map...")
    alias_map = load_alias_map()
    reverse_map = build_reverse_map(alias_map)

    print("\nEvaluating...")
    results = evaluate(cases, alias_map, reverse_map, top_k=args.top_k)

    summary = format_summary(results, test_set_name)
    print(summary)

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
        "--hpo2vec-cache",
        type=Path,
        default=None,
        help="Cache directory for HPO2Vec results (optional)",
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
