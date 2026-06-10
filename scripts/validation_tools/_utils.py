#!/usr/bin/env python3
"""
utils.py — shared utilities for rare disease benchmark pipelines.
Handles loading, YAML generation, TSV parsing, and stats computation.
Designed to be reusable across LIRICAL, Phen2Gene, Exomiser, etc.
"""

import json
import csv
import statistics
from pathlib import Path
from typing import TextIO

# ── Dataset Discovery ──────────────────────────────────────────────────────────


def discover_datasets(data_dir: Path) -> list[str]:
    """
    Discover available datasets by scanning for JSON files in data_dir.

    Returns:
        Sorted list of dataset name stems (e.g. ['HMS', 'MME', 'RAMEDIS']).
    """
    return sorted(p.stem for p in data_dir.glob("*.json"))


def resolve_datasets(data_dir: Path, requested: list[str] | None) -> list[str]:
    """
    Resolve the final list of datasets to run.

    If requested is None, returns all datasets found in data_dir.
    If requested is provided, validates each name against what exists and
    warns about any that are missing.

    Args:
        data_dir:  Directory to scan for JSON files.
        requested: Names passed via --datasets, or None for auto-discover.

    Returns:
        List of valid dataset name stems to process.
    """
    available = discover_datasets(data_dir)

    if not available:
        print(f"[WARN] No JSON datasets found in {data_dir}")
        return []

    if requested is None:
        print(
            f"No --datasets specified. Running all {len(available)} found in {data_dir}:"
        )
        for name in available:
            print(f"  - {name}")
        return available

    resolved = []
    for name in requested:
        match = next((a for a in available if a.lower() == name.lower()), None)
        if match:
            resolved.append(match)
        else:
            print(
                f"[WARN] Requested dataset '{name}' not found in {data_dir}, skipping."
            )
            print(f"  Available datasets: {', '.join(available)}")
    return resolved


# ── JSON loading ──────────────────────────────────────────────────────────────
def load_cases(json_path: Path) -> list[tuple[list[str], list[str]]]:
    """
    Load cases from a JSON file.

    Expected format:
        [ [[hpo_id, ...], [disease_id, ...]], ... ]

    Returns:
        List of (hpo_ids, confirmed_disease_ids) tuples.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = []
    for entry in raw:
        hpo_ids, disease_ids = entry[0], entry[1]
        if not isinstance(hpo_ids, list) or not isinstance(disease_ids, list):
            raise ValueError(f"Unexpected case format in {json_path}: {entry}")
        cases.append((hpo_ids, disease_ids))
    return cases


def load_all_datasets(
    data_dir: Path, names: list[str] | None = None
) -> dict[str, list]:
    """
    Load the given dataset names from data_dir.

    Args:
        data_dir: Directory containing dataset JSON files.
        names:    List of dataset stems to load (already resolved by resolve_datasets).

    Returns:
        Dict mapping dataset name → list of (hpo_ids, disease_ids) tuples.
    """
    datasets = {}

    if names is None:
        names = discover_datasets(data_dir)
        print(f"Auto-discovered datasets in {data_dir}: {', '.join(names)}")

    for name in names:
        path = data_dir / f"{name}.json"

        if not path.exists():
            path = data_dir / f"{name.lower()}.json"

        if not path.exists():
            print(f"[WARN] Could not find file for dataset '{name}', skipping")
            continue

        cases = load_cases(path)
        print(f"Loaded dataset '{name}': {len(cases)} cases from {path.name}")
        datasets[name.lower()] = cases

    return datasets


# ── Summary I/O ───────────────────────────────────────────────────────────────
def save_summary_tsv(summary: list[dict], out_path: Path) -> None:
    """Write a benchmark summary to a TSV file."""
    if not summary:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)


def load_summary_tsv(tsv_path: Path) -> list[dict]:
    """Load a previously saved benchmark summary TSV."""
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_stats(summary: list[dict], top_ks: list[int] = [1, 3, 5, 10, 20]) -> dict:
    """
    Compute benchmark statistics from a summary list.

    Args:
        summary: List of dicts with at least a 'rank' key (int or None).
        top_ks:  List of k values for top-k recall.

    Returns:
        Dict with keys: n, found, not_found, topk (dict), median_rank, mean_rank.
    """
    n = len(summary)
    ranks = []
    query_times = []
    for s in summary:
        r = s["rank"]
        if r is not None and r != "" and r != "None":
            ranks.append(int(r))

        qt = s["query_time_sec"]
        if qt is not None and qt != "" and qt != "None":
            query_times.append(float(qt))

    topk = {k: sum(1 for r in ranks if r <= k) / n for k in top_ks}
    median_rank = statistics.median(ranks) if ranks else None
    mean_rank = statistics.mean(ranks) if ranks else None
    mean_query_time = statistics.mean(query_times) if query_times else None

    return {
        "n": n,
        "found": len(ranks),
        "not_found": n - len(ranks),
        "topk": topk,
        "median_rank": median_rank,
        "mean_rank": round(mean_rank, 2) if mean_rank else None,
        "mean_query_time": round(mean_query_time, 2) if mean_query_time else None,
    }


def compute_mrr(summary: list[dict]) -> float:
    """Mean Reciprocal Rank across all cases (unranked cases score 0)."""
    n = len(summary)
    if n == 0:
        return 0.0
    total = 0.0
    for s in summary:
        r = s["rank"]
        if r is not None and r != "" and r != "None":
            total += 1.0 / int(r)
    return total / n


def print_stats(
    label: str,
    stats: dict,
    f: TextIO,
) -> None:
    """
    Print benchmark statistics to a file if specified, or to stdout.
    """
    n, found = stats["n"], stats["found"]
    f.write(f"\n  {label}  (n={n}, found={found}/{n})\n")

    for k, v in stats["topk"].items():
        bar = "█" * int(v * 25)
        line = f"    top-{k:<3}  {v:.3f}  {bar}\n"
        f.write(line)
