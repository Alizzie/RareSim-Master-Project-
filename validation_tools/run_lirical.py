#!/usr/bin/env python3
"""
run_lirical.py — Run LIRICAL on benchmark datasets and evaluate results.
Uses utils.py for all file handling and statistics.

Runs LIRICAL via `prioritize` subcommand (no YAML input).

Reference: Mao et al., npj Digital Medicine 2025 (PhenoBrain paper)
  LIRICAL on public test set (MME+HMS+LIRICAL):
    top-3=0.407, top-10=0.560, median rank=6.0
    
    
Usage:
python3 validation_tools/run_lirical.py \
  --lirical-jar "/Users/eli/Documents/Uni/Master Project/Tools/LIRICAL/lirical-cli/target/lirical-cli-2.4.0.jar" \
  --lirical-data ~/lirical-data \
  --skip-existing --datasets "HMS"
"""

import os
import subprocess
import argparse
from pathlib import Path
import csv

from utils import (
    DATASET_NAMES,
    load_all_datasets,
    save_summary_tsv,
    compute_stats,
    print_stats,
)

# ── Reference numbers from paper (Supp Table 7) ───────────────────────────────
PAPER_LIRICAL_PUBLIC = {"top3": 0.407, "top10": 0.560, "median_rank": 6.0}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run LIRICAL on PhenoBrain benchmark datasets"
    )
    p.add_argument(
        "--data-dir",
        help="Directory containing the 6 JSON files",
        default="validation_tools/datasets/PhenoBrainBenchmarkDatasets",
    )
    p.add_argument("--lirical-jar", required=True, help="Path to LIRICAL JAR")
    p.add_argument(
        "--lirical-data", required=True, help="Path to LIRICAL data directory"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="Datasets to run (default: all)",
    )
    p.add_argument(
        "--mindiff",
        type=int,
        default=100,
        help="Min differential diagnoses to report (-m flag)",
    )
    p.add_argument("--java", default="java", help="Path to java executable")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip case if TSV output already exists (resume mode)",
    )
    return p.parse_args()


# ── LIRICAL runner ────────────────────────────────────────────────────────────
def run_lirical_case(
    java,
    jar,
    lirical_data,
    case_id,
    hpo_ids,
    negated_hpo_ids,
    out_dir,
    mindiff,
    skip_existing,
):
    """Run LIRICAL prioritize for a single case. Returns True on success."""
    tsv_out = out_dir / f"{case_id}.tsv"
    if skip_existing and tsv_out.exists():
        return True

    cmd = [
        java,
        "-jar",
        jar,
        "prioritize",
        "-d",
        lirical_data,
        "-p",
        ",".join(hpo_ids),
        "--use-orphanet",
        "-f",
        "tsv",
        "-o",
        str(out_dir),
        "-x",
        case_id,
        "-m",
        str(mindiff),
    ]
    if negated_hpo_ids:
        cmd += ["-n", ",".join(negated_hpo_ids)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [WARN] LIRICAL failed for {case_id}: {r.stderr[-200:]}")
        return False
    return True


# ── Per-dataset pipeline ──────────────────────────────────────────────────────
def run_lirical_all(entries, results_dir, args):
    """Phase 1: run LIRICAL on every case. Returns dict case_id -> bool."""
    status = {}
    for i, (case_id, hpo_ids, _) in enumerate(entries):
        ok = run_lirical_case(
            java=args.java,
            jar=args.lirical_jar,
            lirical_data=args.lirical_data,
            case_id=case_id,
            hpo_ids=hpo_ids,
            negated_hpo_ids=[],
            out_dir=results_dir,
            mindiff=args.mindiff,
            skip_existing=args.skip_existing,
        )
        status[case_id] = ok
        if (i + 1) % 50 == 0:
            print(f"  LIRICAL: {i+1}/{len(entries)} done")
    return status


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
    with open(tsv_path, encoding="utf-8") as f:
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
    Uses first-hit strategy.

    Args:
        results: Parsed LIRICAL TSV rows.
        confirmed_diseases: List of confirmed disease IDs (e.g. ["OMIM:191900", "ORPHA:575"]).

    Returns:
        (rank, matched_disease_id, score) or (None, None, None) if none found.
    """
    confirmed = {d.upper() for d in confirmed_diseases}
    for row in sorted(results, key=lambda r: r["rank"]):
        if row["disease_id"].upper() in confirmed:
            return row["rank"], row["disease_id"], row["post_test_prob"]
    return None, None, None


def collect_results(entries, results_dir, run_status):
    """Phase 2: parse TSV outputs and build summary list."""
    summary = []
    for case_id, hpo_ids, confirmed_diseases in entries:
        tsv_path = results_dir / f"{case_id}.tsv"
        lirical_results = parse_lirical_tsv(tsv_path)
        rank, matched_id, score = find_best_rank(lirical_results, confirmed_diseases)
        summary.append(
            {
                "case_id": case_id,
                "n_hpo": len(hpo_ids),
                "confirmed_diseases": ";".join(confirmed_diseases),
                "rank": rank,
                "matched_id": matched_id,
                "score": score,
                "status": run_status.get(case_id, False),
            }
        )

        print(
            f"  {case_id}: rank={rank}, matched_id={matched_id}, "
            f"n_hpo={len(hpo_ids)}, score={score}, "
            f"status={run_status.get(case_id, False)}"
        )
    return summary


def run_dataset(name, cases, args, workdir):
    """Run the full LIRICAL pipeline for one dataset. Returns summary list."""
    results_dir = workdir / "cache" / name
    os.makedirs(results_dir, exist_ok=True)

    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    run_status = run_lirical_all(entries, results_dir, args)
    summary = collect_results(entries, results_dir, run_status)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Main entry point: parse args, run all datasets, print summary."""
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workdir = Path(os.path.join(script_dir, "lirical_benchmarks"))
    os.makedirs(workdir, exist_ok=True)

    # Load all requested datasets
    all_cases = load_all_datasets(Path(args.data_dir), args.datasets)

    # Run LIRICAL and collect summaries
    all_summaries = {}
    for dataset_name, cases in all_cases.items():
        print(f"Processing dataset: {dataset_name} with {len(cases)} cases")
        summary = run_dataset(dataset_name, cases, args, workdir)
        all_summaries[dataset_name] = summary
        save_summary_tsv(summary, workdir / f"{dataset_name}_summary.tsv")

    # Report per-dataset stats
    print("Generating final summary statistics")
    for dataset_name, summary in all_summaries.items():
        with open(f"{dataset_name}_summary.tsv", "w", encoding="utf-8") as f:
            print_stats(
                dataset_name, compute_stats(summary), f, reference=PAPER_LIRICAL_PUBLIC
            )


if __name__ == "__main__":
    main()
