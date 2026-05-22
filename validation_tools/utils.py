#!/usr/bin/env python3
"""
utils.py — shared utilities for rare disease benchmark pipelines.
Handles loading, YAML generation, TSV parsing, and stats computation.
Designed to be reusable across LIRICAL, Phen2Gene, Exomiser, etc.
"""

import glob
import json
import csv
import statistics
from pathlib import Path
from typing import TextIO

# ── Dataset metadata ──────────────────────────────────────────────────────────

# Datasets from: Mao et al., npj Digital Medicine 2025 (PhenoBrain paper)
DATASET_NAMES = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_L", "PUMCH-ADM"]

# MME + HMS + LIRICAL = "Public test set" as defined in the paper
PUBLIC_TEST_DATASETS = ["MME", "HMS", "LIRICAL"]


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


def find_dataset_json(data_dir: Path, name: str) -> Path | None:
    """
    Locate a dataset JSON file by name, tolerating case differences.
    Tries: <name>.json, <name.lower()>.json
    """
    for candidate in [data_dir / f"{name}.json", data_dir / f"{name.lower()}.json"]:
        if candidate.exists():
            return candidate
    return None


def load_all_datasets(
    data_dir: Path, names: list[str] | None = None
) -> dict[str, list]:
    """
    Load datasets from a directory.
    If names is provided, only load those datasets, otherwise load all JSON files.
    """
    all_paths = {
        Path(p).stem.lower(): Path(p) for p in glob.glob(str(data_dir / "*.json"))
    }

    if not all_paths:
        print("No JSON datasets found in", data_dir)
        return {}

    selected = (
        {n.lower(): all_paths[n.lower()] for n in names if n.lower() in all_paths}
        if names
        else all_paths
    )

    if names:
        for n in names:
            if n.lower() not in all_paths:
                print("Dataset '%s' not found in %s, skipping", n, data_dir)

    datasets = {}
    for name, path in selected.items():
        cases = load_cases(path)
        print(f"Loaded dataset '{name}': {len(cases)} cases from {path.name}")
        datasets[name] = cases

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
    for s in summary:
        r = s["rank"]
        if r is not None and r != "" and r != "None":
            ranks.append(int(r))

    topk = {k: sum(1 for r in ranks if r <= k) / n for k in top_ks}
    median_rank = statistics.median(ranks) if ranks else None
    mean_rank = statistics.mean(ranks) if ranks else None

    return {
        "n": n,
        "found": len(ranks),
        "not_found": n - len(ranks),
        "topk": topk,
        "median_rank": median_rank,
        "mean_rank": round(mean_rank, 2) if mean_rank else None,
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
    reference: dict | None = None,
) -> None:
    n, found = stats["n"], stats["found"]
    f.write(f"\n  {label}  (n={n}, found={found}/{n})\n")

    if reference:
        f.write(f"    {'Metric':<12} {'Result':>8} {'Reference':>12} {'Δ':>8}\n")
        f.write(f"    {'─'*44}\n")

    for k, v in stats["topk"].items():
        bar = "█" * int(v * 25)
        line = f"    top-{k:<3}  {v:.3f}  {bar}\n"
        if reference and f"top{k}" in reference:
            ref_v = reference[f"top{k}"]
            delta = v - ref_v
            line = f"    {'top-'+str(k):<12} {v:>8.3f} {ref_v:>12.3f} {delta:>+8.3f}\n"
        f.write(line)

    mr = stats["median_rank"]
    if reference and "median_rank" in reference:
        f.write(f"    {'median rank':<12} {mr!s:>8} {reference['median_rank']!s:>12}\n")
    else:
        f.write(f"    median rank: {mr}\n")
