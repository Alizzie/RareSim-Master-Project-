"""
RareSim evaluation script.

Reads cached case results and computes rank-based metrics for all methods.

Input:
    outputs/evaluation/<DATASET>/cache/case_*.json

Output:
    outputs/evaluation/<DATASET>/
        ├── <DATASET>_evaluation.json
        ├── <DATASET>_evaluation_summary.txt
        ├── <DATASET>_stats.txt
        └── <DATASET>_summary.tsv

Metrics:
    Recall@1  — fraction of cases where the correct disease was ranked 1st
    Recall@3  — fraction of cases where the correct disease was in top 3
    Recall@5  — fraction of cases where the correct disease was in top 5
    Recall@10 — fraction of cases where the correct disease was in top 10
    Recall@20 — fraction of cases where the correct disease was in top 20
    MRR       — Mean Reciprocal Rank  (1/rank, averaged across cases,
                using the best-ranked ground-truth match per case)
    NDCG@10   — Normalized DCG at cutoff 10, computed over ALL matched
                ground-truth diseases found within the top 10 (not just
                the best-ranked one), normalized against the ideal DCG
                for that case's number of distinct ground-truth diseases.
                Ground-truth IDs that are aliases of the same underlying
                disease (per alias_to_canonical.json) are counted once,
                not once per identifier.
    Median rank — median rank of the correct disease across all found cases

Per-case timing:
    Average seconds per case is reported for each method based on the
    method_elapsed_seconds field written by the batch runner.

Usage:
    python scripts/evaluation/evaluator.py --dataset HMS

Adding a new method
-------------------
The evaluator automatically detects methods from the per-case cache files.

To add a new method:
1. Implement the method runner.
2. Make sure it writes results into:
       outputs/evaluation/<DATASET>/cache/case_XXXX.json
3. Inside each case file, the method must appear under:
       case["results"][METHOD_NAME]
4. Each result should contain at least:
       disease_id or canonical_disease_id or ordo_id
       rank
       score
       label
5. Add timing under:
       case["method_elapsed_seconds"][METHOD_NAME]
6. Add the method name to:
       case["methods_run"]

No evaluator logic needs to change if the result schema follows this format.
"""
# pylint: disable=broad-exception-caught
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-return-statements

import argparse
import csv
import json
import math

from collections import defaultdict
from pathlib import Path
from raresim.utils.io import load_json
from raresim.utils.paths import ALIAS_TO_CANONICAL_PATH
from _batch_utils import EVALUATION_DIR

# ── RRF configuration ─────────────────────────────────────────────────────────

RRF_K = 60
RRF_METHOD_NAME = "ensemble_rrf"
RRF_WEIGHTED_NAME = "ensemble_rrf_weighted"
RRF_TOP_NAME = "ensemble_rrf_top"
RRF_MIN_RECALL10 = 0.10  # minimum R@10 for a method to join the top ensemble


# ── ID normalisation ───────────────────────────────────────────────────────────


def load_alias_map() -> dict[str, str]:
    """Load alias-to-canonical disease ID mappings if available."""
    if ALIAS_TO_CANONICAL_PATH.exists():
        return load_json(ALIAS_TO_CANONICAL_PATH)
    print("[warning] alias_to_canonical.json not found — using direct ID matching only")
    return {}


def build_reverse_map(alias_map: dict[str, str]) -> dict[str, set[str]]:
    """Build a reverse lookup from canonical IDs to all equivalent aliases."""
    reverse: dict[str, set[str]] = defaultdict(set)
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
    """Return all IDs equivalent to one disease ID according to alias maps."""
    ids = {disease_id}
    canonical = alias_map.get(disease_id, disease_id)
    ids.add(canonical)
    ids.update(reverse_map.get(disease_id, set()))
    ids.update(reverse_map.get(canonical, set()))
    return ids


def count_distinct_ground_truth(
    ground_truth_ids: list[str],
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
) -> int:
    """
    Count ground-truth diseases as distinct alias-equivalence groups, not a
    raw ID count. Two ground-truth identifiers that are aliases of the same
    underlying disease (e.g. an OMIM/ORPHA cross-reference pair) count as one
    relevant item, not two -- otherwise the ideal-DCG normalization in NDCG
    would be based on an inflated number of "relevant" diseases and unfairly
    deflate the score for a case that only has one true diagnosis.
    """
    groups: list[set[str]] = []
    for gt_id in ground_truth_ids:
        eq = get_all_equivalent_ids(gt_id, alias_map, reverse_map)
        merged = False
        for group in groups:
            if group & eq:
                group |= eq
                merged = True
                break
        if not merged:
            groups.append(eq)
    return len(groups)


# ── Result loading ─────────────────────────────────────────────────────────────


def get_disease_id_from_result(result: dict) -> str | None:
    """
    Extract a disease ID from a result dict, regardless of pipeline schema.

    - semantic / set-based / tfidf : 'disease_id'
    - transformer                  : 'canonical_disease_id'
    - llm                          : 'ordo_id'
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
    Return the best rank (1-indexed) of any ground-truth disease in results,
    or None if not found. Used for Recall@k, MRR, and median rank, which are
    single-best-hit metrics by definition.
    """
    gt_equivalent: set[str] = set()
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


def find_all_matched_ranks(
    ground_truth_ids: list[str],
    results: list[dict],
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
) -> list[int]:
    """
    Return the ranks (1-indexed) of every distinct candidate in results that
    matches any ground-truth disease, not just the first/best one. Used for
    true multi-relevance NDCG, where credit should be given for every
    correct disease found in the top k, not only the highest-ranked one.

    Because results is a list of distinct candidate diseases (one entry per
    rank), and matching is done against the union of all ground-truth
    equivalence sets, a ground-truth pair that are themselves aliases of one
    disease cannot cause the same candidate to be counted twice here.
    """
    gt_equivalent: set[str] = set()
    for gt_id in ground_truth_ids:
        gt_equivalent.update(get_all_equivalent_ids(gt_id, alias_map, reverse_map))

    matched_ranks: list[int] = []
    for result in results:
        result_id = get_disease_id_from_result(result)
        if not result_id:
            continue
        result_equivalent = get_all_equivalent_ids(result_id, alias_map, reverse_map)
        if gt_equivalent & result_equivalent:
            rank = result.get("rank")
            if rank is not None:
                matched_ranks.append(rank)

    return sorted(matched_ranks)


def load_cache_dir(cache_dir: Path) -> list[dict]:
    """Load all case_*.json files from cache_dir, sorted by index."""
    if not cache_dir.exists():
        return []
    cases = []
    for f in sorted(cache_dir.glob("case_*.json")):
        try:
            with f.open(encoding="utf-8") as fp:
                cases.append(json.load(fp))
        except Exception as e:
            print(f"[warning] Could not load {f.name}: {e}")
    return cases


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_ndcg_for_case(
    matched_ranks: list[int],
    n_relevant: int,
    top_k: int = 10,
) -> float:
    """
    True NDCG@k for one case, with binary relevance across ALL matched
    ground-truth diseases found in the top k -- not just the best-ranked one.

        DCG@k  = sum over matched ranks r <= k of  1 / log2(r + 1)
        IDCG@k = the DCG of the ideal ordering, i.e. all n_relevant ground-
                 truth diseases placed at ranks 1..min(n_relevant, k)
        NDCG@k = DCG@k / IDCG@k   (0 if IDCG@k is 0, i.e. no ground truth)

    n_relevant should be the number of DISTINCT ground-truth diseases for
    this case (see count_distinct_ground_truth), not the raw length of the
    ground_truth list, since two IDs that are aliases of the same disease
    are only one relevant item.
    """
    dcg = sum(1.0 / math.log2(r + 1) for r in matched_ranks if r <= top_k)

    ideal_hits = min(n_relevant, top_k)
    idcg = sum(1.0 / math.log2(j + 1) for j in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_metrics(
    ranks: list[int | None],
    ndcg_values: list[float],
    top_k: int = 10,
) -> dict:
    """
    Compute Recall@1/3/5/10/20, MRR, NDCG@10, median rank from per-case data.

    ranks       : per-case best rank (or None if not found) -- drives
                  Recall@k, MRR, and median rank, which are single-best-hit
                  metrics by definition.
    ndcg_values : per-case NDCG@top_k value, precomputed by the caller via
                  compute_ndcg_for_case() using ALL matched ground-truth
                  diseases for that case, not just the best rank.
    """
    n = len(ranks)
    if n == 0:
        return {
            "recall_1": 0,
            "recall_3": 0,
            "recall_5": 0,
            "recall_10": 0,
            "recall_20": 0,
            "mrr": 0,
            "ndcg": 0,
            "found": 0,
            "median_rank": None,
        }

    found_ranks = sorted(r for r in ranks if r is not None)
    median_rank = found_ranks[len(found_ranks) // 2] if found_ranks else None

    return {
        "recall_1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "recall_3": round(sum(1 for r in ranks if r is not None and r <= 3) / n, 4),
        "recall_5": round(sum(1 for r in ranks if r is not None and r <= 5) / n, 4),
        "recall_10": round(sum(1 for r in ranks if r is not None and r <= 10) / n, 4),
        "recall_20": round(sum(1 for r in ranks if r is not None and r <= 20) / n, 4),
        "mrr": round(sum(1 / r for r in ranks if r is not None) / n, 4),
        "ndcg": round(sum(ndcg_values) / n, 4),
        "found": len(found_ranks),
        "median_rank": median_rank,
    }


# ── Timing helpers ─────────────────────────────────────────────────────────────


def aggregate_method_timing(cases: list[dict]) -> dict[str, float]:
    """
    Return average elapsed seconds per method across all cases that
    reported timing for that method.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        for method, elapsed in case.get("method_elapsed_seconds", {}).items():
            totals[method] += elapsed
            counts[method] += 1
    return {
        method: round(totals[method] / counts[method], 3)
        for method in totals
        if counts[method] > 0
    }


# ── RRF ensemble ──────────────────────────────────────────────────────────────


def compute_rrf(
    case_results: dict[str, list[dict]],
    methods: list[str],
    top_k: int = 10,
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion for a single case.

        RRF(d) = sum_{r in R}  w_r / (k + rank_r(d))

    weights=None → equal weighting (w_r = 1.0).
    Returns top_k results as [{'disease_id': ..., 'rank': ...}].
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for method in methods:
        w = weights[method] if weights else 1.0
        if w == 0:
            continue
        for result in case_results.get(method, []):
            disease_id = get_disease_id_from_result(result)
            rank = result.get("rank")
            if disease_id and rank is not None:
                rrf_scores[disease_id] += w / (k + rank)

    return [
        {"disease_id": did, "rank": new_rank}
        for new_rank, (did, _) in enumerate(
            sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k], start=1
        )
    ]


# ── Core evaluation ────────────────────────────────────────────────────────────


def evaluate(
    cases: list[dict],
    alias_map: dict[str, str],
    reverse_map: dict[str, set[str]],
    top_k: int = 10,
) -> dict:
    """
    Evaluate all methods across all cases.

    Three RRF ensemble variants are produced:
      ensemble_rrf          — equal weights, all base methods
      ensemble_rrf_weighted — weighted by each method's R@10
      ensemble_rrf_top      — equal weights, only methods with R@10 >= threshold
    """
    # Collect the full set of base methods present in the cache
    base_methods: set[str] = set()
    for case in cases:
        base_methods.update(case.get("results", {}).keys())
    base_methods_sorted = sorted(base_methods)

    all_methods = base_methods_sorted + [
        RRF_METHOD_NAME,
        RRF_WEIGHTED_NAME,
        RRF_TOP_NAME,
    ]
    method_ranks: dict[str, list[int | None]] = {m: [] for m in all_methods}
    method_ndcg: dict[str, list[float]] = {m: [] for m in all_methods}

    # Ground-truth cardinality is a case-level property (same for every
    # method), so it only needs to be computed once per case.
    n_relevant_per_case: dict[int, int] = {}
    for case in cases:
        n_relevant_per_case[case["case_index"]] = count_distinct_ground_truth(
            case.get("ground_truth", []), alias_map, reverse_map
        )

    # ── Pass 1: per-method ranks and NDCG ─────────────────────────────────────
    rank_matrix_pass1 = []
    for case in cases:
        ground_truth = case.get("ground_truth", [])
        results = case.get("results", {})
        n_relevant = n_relevant_per_case[case["case_index"]]
        case_ranks: dict[str, int | None] = {}
        for method in base_methods_sorted:
            method_results = results.get(method, [])
            rank = find_rank(ground_truth, method_results, alias_map, reverse_map)
            matched_ranks = find_all_matched_ranks(
                ground_truth, method_results, alias_map, reverse_map
            )
            method_ranks[method].append(rank)
            method_ndcg[method].append(
                compute_ndcg_for_case(matched_ranks, n_relevant, top_k)
            )
            case_ranks[method] = rank
        rank_matrix_pass1.append((case, case_ranks))

    # ── Compute R@10 weights for RRF ──────────────────────────────────────────
    n = len(cases)
    method_recall10 = {
        method: sum(1 for r in method_ranks[method] if r is not None and r <= 10) / n
        for method in base_methods_sorted
    }
    top_methods = [
        m for m in base_methods_sorted if method_recall10[m] >= RRF_MIN_RECALL10
    ]
    if not top_methods:
        top_methods = base_methods_sorted  # fallback: use all

    # ── Pass 2: RRF ensemble variants ─────────────────────────────────────────
    rank_matrix = []
    for case, case_ranks in rank_matrix_pass1:
        ground_truth = case.get("ground_truth", [])
        results = case.get("results", {})
        n_relevant = n_relevant_per_case[case["case_index"]]

        for ensemble_name, methods, weights in [
            (RRF_METHOD_NAME, base_methods_sorted, None),
            (RRF_WEIGHTED_NAME, base_methods_sorted, method_recall10),
            (RRF_TOP_NAME, top_methods, None),
        ]:
            rrf_results = compute_rrf(results, methods, top_k, weights=weights)
            rank = find_rank(ground_truth, rrf_results, alias_map, reverse_map)
            matched_ranks = find_all_matched_ranks(
                ground_truth, rrf_results, alias_map, reverse_map
            )
            method_ranks[ensemble_name].append(rank)
            method_ndcg[ensemble_name].append(
                compute_ndcg_for_case(matched_ranks, n_relevant, top_k)
            )
            case_ranks[ensemble_name] = rank

        rank_matrix.append(
            {
                "case_index": case["case_index"],
                "ground_truth": ground_truth,
                "ranks": case_ranks,
            }
        )

    method_metrics = {
        m: compute_metrics(method_ranks[m], method_ndcg[m], top_k) for m in all_methods
    }
    avg_timing = aggregate_method_timing(cases)

    return {
        "n_cases": n,
        "n_methods": len(all_methods),
        "methods": all_methods,
        "method_metrics": method_metrics,
        "method_avg_seconds": avg_timing,
        "rank_matrix": rank_matrix,
        "rrf_top_methods": top_methods,
        "rrf_method_weights": method_recall10,
    }


# ── Method agreement analysis ──────────────────────────────────────────────────


def compute_agreement(results: dict) -> dict:
    """
    Compute consensus / hard / easy / unique-find statistics across all cases.
    """
    rank_matrix = results["rank_matrix"]
    all_methods = results["methods"]
    n_methods = len(all_methods)

    consensus_count = unique_find_count = hard_count = easy_count = 0
    found_by_n: dict[int, int] = defaultdict(int)
    rank_histogram: dict[int, int] = defaultdict(int)
    unique_finds: list[dict] = []
    hard_cases: list[str] = []

    for row in rank_matrix:
        case_id = f"case_{row['case_index']:04d}"
        ranks = row["ranks"]

        found_methods = [m for m in all_methods if ranks.get(m) is not None]
        n_found = len(found_methods)
        found_by_n[n_found] += 1

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
            solo = found_methods[0]
            unique_finds.append(
                {
                    "case_id": case_id,
                    "gt": row["ground_truth"],
                    "method": solo,
                    "rank": ranks[solo],
                }
            )

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


# ── Formatting ─────────────────────────────────────────────────────────────────

_BAR_FULL = "█"
_BAR_WIDTH = 40


def _bar(value: int, max_value: int) -> str:
    """Return a fixed-width text bar for integer counts."""
    if max_value == 0:
        return ""
    filled = round(_BAR_WIDTH * value / max_value)
    return _BAR_FULL * filled


def format_summary(results: dict, test_set_name: str) -> str:
    """Format the complete evaluation summary as plain text."""
    n = results["n_cases"]
    metrics = results["method_metrics"]
    rank_matrix = results["rank_matrix"]
    avg_timing = results.get("method_avg_seconds", {})

    lines: list[str] = []
    w = 80

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"\n{'=' * w}",
        f"  Evaluation Summary — {test_set_name}  ({n} cases)",
        f"{'=' * w}",
    ]

    # ── Method comparison table ───────────────────────────────────────────────
    lines += [
        f"\n{'=' * w}",
        "  Method Comparison — Recall@k, MRR, NDCG@10, Avg time/case",
        f"{'=' * w}",
        "  "
        f"{'Method':<45} {'R@1':>6} {'R@3':>6} {'R@5':>6} "
        f"{'R@10':>6} {'R@20':>6} {'MRR':>6} {'NDCG':>6}  "
        f"{'Found':<10} {'Avg(s)':>7}",
        f"  {'-' * (w - 2)}",
    ]

    sorted_methods = sorted(
        metrics.items(),
        key=lambda x: (-x[1]["recall_10"], -x[1]["mrr"]),
    )
    for method, m in sorted_methods:
        avg_s = avg_timing.get(method)
        avg_str = f"{avg_s:.3f}" if avg_s is not None else "  n/a"
        lines.append(
            f"  {method:<45} "
            f"{m['recall_1']:>6.4f} "
            f"{m.get('recall_3', 0):>6.4f} "
            f"{m['recall_5']:>6.4f} "
            f"{m['recall_10']:>6.4f} "
            f"{m.get('recall_20', 0):>6.4f} "
            f"{m['mrr']:>6.4f} "
            f"{m['ndcg']:>6.4f}  "
            f"{m['found']}/{n:<6}  "
            f"{avg_str:>7}"
        )
    lines.append(f"{'=' * w}")

    # ── RRF ensemble configuration ────────────────────────────────────────────
    top_methods = results.get("rrf_top_methods", [])
    rrf_weights = results.get("rrf_method_weights", {})
    n_base = results["n_methods"] - 3  # subtract the 3 ensemble methods

    lines += [
        f"\n{'=' * w}",
        "  RRF Ensemble Configuration",
        f"{'=' * w}",
        f"  Threshold  : R@10 >= {RRF_MIN_RECALL10}  for {RRF_TOP_NAME}",
        f"  Top methods: {len(top_methods)} / {n_base} base methods",
    ]
    for m in top_methods:
        lines.append(f"    {m:<45} R@10={rrf_weights.get(m, 0):.4f}")
    lines.append(f"{'=' * w}")

    # ── Method agreement ──────────────────────────────────────────────────────
    agreement = compute_agreement(results)
    lines.append(format_agreement_section(agreement, n))

    # ── Per-case rank matrix ──────────────────────────────────────────────────
    all_methods = results["methods"]
    lines += [
        f"\n{'=' * w}",
        "  Per-Case Rank Matrix  (- = not found in top 10)",
        f"{'=' * w}",
    ]
    method_headers = "  ".join(f"{m[:12]:>12}" for m in all_methods)
    lines.append(f"  {'Case':<12} {'GT':<25}  {method_headers}")
    lines.append(f"  {'-' * (w - 2)}")

    for row in rank_matrix:
        case_id = f"case_{row['case_index']:04d}"
        gt_str = ",".join(row["ground_truth"])[:24]
        rank_strs = "  ".join(
            f"{str(row['ranks'].get(m) or '-'):>12}" for m in all_methods
        )
        lines.append(f"  {case_id:<12} {gt_str:<25}  {rank_strs}")

    lines.append(f"{'=' * w}")
    return "\n".join(lines)


def format_agreement_section(agreement: dict, n_cases: int) -> str:
    """Format the method-agreement section as plain text."""
    lines: list[str] = []
    w = 72
    lines += [
        f"\n{'=' * w}",
        "  Method Agreement Analysis",
        f"{'=' * w}",
        f"  Total cases : {n_cases}",
        f"  Consensus   : {agreement['consensus_count']} cases — all methods found it",
        f"  Hard cases  : {agreement['hard_count']} cases — no method found it",
        f"  Easy cases  : {agreement['easy_count']} cases — at least one method ranked it #1",
        f"  Unique finds: {agreement['unique_find_count']} cases — only one method found it",
        "\n  How many methods found the correct disease per case:",
    ]
    found_by_n = agreement["found_by_n"]
    n_methods = agreement["n_methods"]
    max_val = max(found_by_n.values(), default=1)
    for k in range(n_methods + 1):
        count = found_by_n.get(k, 0)
        lines.append(f"    {k} methods: {count:>3} cases  {_bar(count, max_val)}")

    lines.append("\n  How many methods found correct disease at each rank:")
    rank_hist = agreement["rank_histogram"]
    if rank_hist:
        max_rank = max(rank_hist.keys())
        max_hist_val = max(rank_hist.values(), default=1)
        for r in range(1, max_rank + 1):
            count = rank_hist.get(r, 0)
            lines.append(f"    rank_{r:>2}: {count:>3}  {_bar(count, max_hist_val)}")

    lines.append("\n  Unique finds (only one method found it):")
    if agreement["unique_finds"]:
        for uf in agreement["unique_finds"]:
            lines.append(
                f"    {uf['case_id']} | gt={uf['gt']} | {uf['method']} @ rank {uf['rank']}"
            )
    else:
        lines.append("    (none)")

    lines.append("\n  Hard cases (no method found the correct disease):")
    if agreement["hard_cases"]:
        for hc in agreement["hard_cases"]:
            lines.append(f"    {hc}")
    else:
        lines.append("    (none)")

    lines.append(f"{'=' * w}")
    return "\n".join(lines)


# ── Unified output formatters (all methods in one file each) ──────────────────

_BAR_FULL_COMPAT = "█"
_BAR_WIDTH_COMPAT = 20


def _compat_bar(value: float) -> str:
    """Return a compact text bar for a 0-1 metric value."""
    filled = round(_BAR_WIDTH_COMPAT * value)
    return _BAR_FULL_COMPAT * filled


def write_stats_txt(results: dict, path: Path) -> None:
    """Write a single _stats.txt with all methods stacked, sorted by R@10."""
    n = results["n_cases"]
    avg_timing = results.get("method_avg_seconds", {})
    metrics = results["method_metrics"]

    sorted_methods = sorted(
        metrics.items(),
        key=lambda x: (-x[1]["recall_10"], -x[1]["mrr"]),
    )

    blocks = []
    for method, m in sorted_methods:
        avg_s = avg_timing.get(method)
        lines = [
            f"{method}  (n={n}, found={m['found']}/{n})",
            f"    top-1   {m['recall_1']:.3f}  {_compat_bar(m['recall_1'])}",
            f"    top-3   {m['recall_3']:.3f}  {_compat_bar(m['recall_3'])}",
            f"    top-5   {m['recall_5']:.3f}  {_compat_bar(m['recall_5'])}",
            f"    top-10  {m['recall_10']:.3f}  {_compat_bar(m['recall_10'])}",
            f"    top-20  {m['recall_20']:.3f}  {_compat_bar(m['recall_20'])}",
            f"    median rank: {m['median_rank']}",
        ]
        if avg_s is not None:
            lines.append(f"    avg query time: {avg_s:.3f}s")
        blocks.append("\n".join(lines))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_summary_tsv(results: dict, cases: list[dict], path: Path) -> None:
    """Write a single _summary.tsv with all methods, one row per case per method."""
    rank_matrix = {row["case_index"]: row for row in results["rank_matrix"]}
    all_methods = results["methods"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "method",
                "case_id",
                "n_hpo",
                "confirmed_diseases",
                "rank",
                "matched_id",
                "status",
                "query_time_sec",
            ]
        )
        for case in cases:
            idx = case["case_index"]
            case_id = f"case_{idx:04d}"
            n_hpo = len(case.get("hpo_terms", []))
            confirmed = ";".join(case.get("ground_truth", []))
            row_data = rank_matrix.get(idx, {})

            for method in all_methods:
                rank = row_data.get("ranks", {}).get(method)
                matched_id = "None"
                if rank is not None:
                    for r in case.get("results", {}).get(method, []):
                        if r.get("rank") == rank:
                            matched_id = (
                                r.get("disease_id")
                                or r.get("canonical_disease_id")
                                or r.get("ordo_id")
                                or "None"
                            )
                            break
                query_time = case.get("method_elapsed_seconds", {}).get(method, "None")
                writer.writerow(
                    [
                        method,
                        case_id,
                        n_hpo,
                        confirmed,
                        rank if rank is not None else "None",
                        matched_id,
                        rank is not None,
                        (
                            f"{query_time:.3f}"
                            if isinstance(query_time, float)
                            else query_time
                        ),
                    ]
                )


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="RareSim evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="dataset",
        required=True,
        help="Dataset name, e.g. HMS, MME, RAMEDIS",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff for Recall@k and NDCG (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation for one cached dataset."""
    args = parse_args()
    test_set_name = str(args.dataset)
    dataset_dir = EVALUATION_DIR / test_set_name
    cache_dir = dataset_dir / "cache"

    print(f"\nLoading cached results from {cache_dir} ...")
    cases = load_cache_dir(cache_dir)
    print(f"  Loaded {len(cases)} cases.")

    if not cases:
        print("No cases found — nothing to evaluate.")
        return

    method_coverage: dict[str, int] = defaultdict(int)
    for case in cases:
        for method in case.get("methods_run", []):
            method_coverage[method] += 1

    print(f"\n  Method coverage ({len(cases)} cases total):")
    for method, count in sorted(method_coverage.items()):
        print(f"    {method:<45} {count}/{len(cases)} cases cached")

    print("\nLoading alias map ...")
    alias_map = load_alias_map()
    reverse_map = build_reverse_map(alias_map)

    print("\nEvaluating ...")
    results = evaluate(cases, alias_map, reverse_map, top_k=args.top_k)

    summary = format_summary(results, test_set_name)
    print(summary)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    json_path = dataset_dir / f"{test_set_name}_evaluation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON : {json_path}")

    summary_path = dataset_dir / f"{test_set_name}_evaluation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved TXT  : {summary_path}")

    stats_path = dataset_dir / f"{test_set_name}_stats.txt"
    write_stats_txt(results, stats_path)
    print(f"Saved TXT  : {stats_path}")

    tsv_path = dataset_dir / f"{test_set_name}_summary.tsv"
    write_summary_tsv(results, cases, tsv_path)
    print(f"Saved TSV  : {tsv_path}")


if __name__ == "__main__":
    main()
