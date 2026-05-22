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
    with open(json_path) as f:
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


# ── YAML generation ───────────────────────────────────────────────────────────


def write_lirical_yaml(
    case_id: str,
    hpo_ids: list[str],
    out_path: Path,
    negated_hpo_ids: list[str] | None = None,
) -> Path:
    """
    Write a LIRICAL-compatible YAML input file for a single case.

    Returns:
        The path the file was written to.
    """
    negated = negated_hpo_ids or []
    lines = [
        "---",
        f"sampleId: {case_id}",
        "hpoIds:",
    ]
    for hpo in hpo_ids:
        lines.append(f"  - {hpo}")
    lines.append("negatedHpoIds:")
    for hpo in negated:
        lines.append(f"  - {hpo}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_yaml_batch(cases: list[tuple], prefix: str, yaml_dir: Path) -> list[Path]:
    """
    Write YAML files for a list of (hpo_ids, disease_ids) cases.

    Returns:
        List of written YAML paths (same order as cases).
    """
    yaml_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (hpo_ids, _) in enumerate(cases):
        case_id = f"{prefix}_case_{i:04d}"
        path = yaml_dir / f"{case_id}.yaml"
        write_lirical_yaml(case_id, hpo_ids, path)
        paths.append(path)
    return paths


# ── TSV parsing ───────────────────────────────────────────────────────────────


def parse_lirical_tsv(tsv_path: Path) -> list[dict]:
    """
    Parse a LIRICAL TSV output file.

    Returns:
        List of dicts with keys: rank, disease_id, disease_name, post_test_prob.
        Empty list if file does not exist.
    """
    if not tsv_path.exists():
        return []
    rows = []
    with open(tsv_path) as f:
        # Skip comment lines (start with '!')
        lines = [l for l in f if not l.startswith("!")]
    reader = csv.DictReader(lines, delimiter="\t")
    for row in reader:
        disease_id = (
            row.get("diseaseCurie")
            or row.get("diseaseId")
            or row.get("disease_id")
            or ""
        ).strip()
        rows.append(
            {
                "rank": int(str(row.get("rank", 0)).replace(",", "")),
                "disease_id": disease_id,
                "disease_name": (
                    row.get("diseaseName") or row.get("disease_name") or ""
                ).strip(),
                "post_test_prob": (
                    row.get("posttestprob") or row.get("postTestProbability") or ""
                ).strip(),
            }
        )
    return rows


def find_best_rank(
    results: list[dict], confirmed_diseases: list[str]
) -> tuple[int | None, str | None]:
    """
    Find the best (lowest) rank among all confirmed disease IDs.
    Uses first-hit strategy (matching paper methodology).

    Args:
        results: Parsed LIRICAL TSV rows.
        confirmed_diseases: List of confirmed disease IDs (e.g. ["OMIM:191900", "ORPHA:575"]).

    Returns:
        (rank, matched_disease_id) or (None, None) if none found.
    """
    confirmed = {d.upper() for d in confirmed_diseases}
    for row in sorted(results, key=lambda r: r["rank"]):
        if row["disease_id"].upper() in confirmed:
            return row["rank"], row["disease_id"]
    return None, None


# ── Summary I/O ───────────────────────────────────────────────────────────────


def save_summary_tsv(summary: list[dict], out_path: Path) -> None:
    """Write a benchmark summary to a TSV file."""
    if not summary:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)


def load_summary_tsv(tsv_path: Path) -> list[dict]:
    """Load a previously saved benchmark summary TSV."""
    with open(tsv_path) as f:
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


def print_stats(label: str, stats: dict, reference: dict | None = None) -> None:
    """
    Pretty-print benchmark statistics, optionally comparing to a reference.

    Args:
        label:     Display name for the dataset.
        stats:     Output of compute_stats().
        reference: Optional dict with keys top3, top10, median_rank for comparison.
    """
    n, found = stats["n"], stats["found"]
    print(f"\n  {label}  (n={n}, found={found}/{n})")

    if reference:
        print(f"    {'Metric':<12} {'Result':>8} {'Reference':>12} {'Δ':>8}")
        print(f"    {'─'*44}")

    for k, v in stats["topk"].items():
        bar = "█" * int(v * 25)
        line = f"    top-{k:<3}  {v:.3f}  {bar}"
        if reference and f"top{k}" in reference:
            ref_v = reference[f"top{k}"]
            delta = v - ref_v
            line = f"    {'top-'+str(k):<12} {v:>8.3f} {ref_v:>12.3f} {delta:>+8.3f}"
        print(line)

    mr = stats["median_rank"]
    if reference and "median_rank" in reference:
        print(f"    {'median rank':<12} {mr!s:>8} {reference['median_rank']!s:>12}")
    else:
        print(f"    median rank: {mr}")
