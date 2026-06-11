"""
Post-processing summary of all pipeline results:
- Disease occurrence across methods with ranks and scores
- Computation time per method
"""

from collections import defaultdict
from raresim.types.result import MethodResults

# ── Disease co-occurrence ─────────────────────────────────────────────────────


def build_disease_summary(
    all_results: dict[str, MethodResults],
) -> list[dict]:
    """
    For each disease that appears in any method's top-k results,
    collect how many methods ranked it, with ranks and scores per method.

    Returns a list sorted by:
      1. n_methods_ranked (descending) — appeared in most methods first
      2. avg_score (descending) — higher average score first
    """
    # disease_id → {label, methods: {method_name: {rank, score}}}
    disease_map: dict[str, dict] = defaultdict(lambda: {"label": "", "methods": {}})

    for method_name, method_results in all_results.items():
        for r in method_results.rankings:
            entry = disease_map[r.disease_id]
            entry["label"] = r.label
            entry["methods"][method_name] = {
                "rank": r.rank,
                "score": round(r.score, 4),
            }

    summary = []
    for disease_id, data in disease_map.items():
        scores = [m["score"] for m in data["methods"].values()]
        ranks = [m["rank"] for m in data["methods"].values()]
        summary.append(
            {
                "disease_id": disease_id,
                "label": data["label"],
                "n_methods_ranked": len(data["methods"]),
                "avg_score": round(sum(scores) / len(scores), 4),
                "avg_rank": round(sum(ranks) / len(ranks), 2),
                "best_rank": min(ranks),
                "methods": dict(
                    sorted(data["methods"].items(), key=lambda x: x[1]["rank"])
                ),
            }
        )

    return sorted(
        summary,
        key=lambda x: (-x["n_methods_ranked"], -x["avg_score"]),
    )


# ── Computation times ─────────────────────────────────────────────────────────


def build_timing_summary(
    all_results: dict[str, MethodResults],
) -> list[dict]:
    """
    Returns computation time per method, sorted slowest to fastest.
    """
    timings = []
    for method_name, method_results in all_results.items():
        timings.append(
            {
                "method_name": method_name,
                "computation_time_s": round(
                    method_results.metadata.computation_time, 3
                ),
            }
        )

    return sorted(timings, key=lambda x: x["computation_time_s"], reverse=True)


# ── Display ───────────────────────────────────────────────────────────────────


def print_disease_summary(summary: list[dict], top_n: int = 20) -> None:
    print(f"\n{'─' * 64}")
    print("  Disease co-occurrence across methods")
    print(f"{'─' * 64}")
    print(
        f"  {'Disease ID':<16} {'Methods':>7} {'Avg Rank':>9} "
        f"{'Avg Score':>10}  Label"
    )
    print(f"  {'─' * 62}")

    for entry in summary[:top_n]:
        method_detail = "  ".join(
            f"{name}(#{info['rank']}={info['score']})"
            for name, info in entry["methods"].items()
        )
        print(
            f"  {entry['disease_id']:<16} "
            f"{entry['n_methods_ranked']:>7} "
            f"{entry['avg_rank']:>9.1f} "
            f"{entry['avg_score']:>10.4f}  "
            f"{entry['label']}"
        )
        print(f"    └─ {method_detail}")


def print_timing_summary(timings: list[dict]) -> None:
    print(f"\n{'─' * 64}")
    print("  Computation time per method")
    print(f"{'─' * 64}")
    total = sum(t["computation_time_s"] for t in timings)
    for t in timings:
        bar_len = int((t["computation_time_s"] / total) * 30) if total > 0 else 0
        bar = "█" * bar_len
        print(f"  {t['method_name']:<30} " f"{t['computation_time_s']:>6.3f}s  {bar}")
    print(f"  {'─' * 40}")
    print(f"  {'total':<30} {total:>6.3f}s")
