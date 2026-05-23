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
import time
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
) -> tuple[bool, float | None]:
    """Run LIRICAL prioritize for a single case. Returns (success, query_time_sec)."""
    tsv_out = out_dir / f"{case_id}.tsv"
    if skip_existing and tsv_out.exists():
        return True, None  # timing not available for cached results

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

    start_time = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    query_time_sec = time.time() - start_time

    if r.returncode != 0:
        print(f"    [WARN] LIRICAL failed for {case_id}: {r.stderr[-200:]}")
        return False, query_time_sec
    return True, query_time_sec


# ── Per-dataset pipeline ──────────────────────────────────────────────────────
def run_lirical_all(entries, results_dir, args) -> tuple[dict, dict]:
    """Phase 1: run LIRICAL on every case. Returns (status, request_times) dicts."""
    status = {}
    request_times = {}
    for i, (case_id, hpo_ids, _) in enumerate(entries):
        ok, query_time_sec = run_lirical_case(
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
        request_times[case_id] = query_time_sec
        if (i + 1) % 50 == 0:
            print(f"  LIRICAL: {i+1}/{len(entries)} done")
    return status, request_times


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
) -> tuple[int | None, str | None, str | None]:
    """Find the best (lowest) rank among all confirmed disease IDs."""
    confirmed = {d.upper() for d in confirmed_diseases}
    for row in sorted(results, key=lambda r: r["rank"]):
        if row["disease_id"].upper() in confirmed:
            return row["rank"], row["disease_id"], row["post_test_prob"]
    return None, None, None


def collect_results(entries, results_dir, run_status, request_times) -> list[dict]:
    """Phase 2: parse TSV outputs and build summary list."""
    summary = []
    for case_id, hpo_ids, confirmed_diseases in entries:
        tsv_path = results_dir / f"{case_id}.tsv"
        lirical_results = parse_lirical_tsv(tsv_path)
        rank, matched_id, score = find_best_rank(lirical_results, confirmed_diseases)
        query_time_sec = request_times.get(case_id)
        summary.append(
            {
                "case_id": case_id,
                "n_hpo": len(hpo_ids),
                "confirmed_diseases": ";".join(confirmed_diseases),
                "rank": rank,
                "matched_id": matched_id,
                "score": score,
                "status": run_status.get(case_id, False),
                "query_time_sec": (
                    f"{query_time_sec:.3f}" if query_time_sec is not None else "None"
                ),
            }
        )
        print(
            f"  {case_id}: rank={rank}, matched_id={matched_id}, "
            f"n_hpo={len(hpo_ids)}, score={score}, "
            f"status={run_status.get(case_id, False)}, query_time_sec={query_time_sec}"
        )
    return summary


def run_dataset(name, cases, args, workdir) -> list[dict]:
    """Run the full LIRICAL pipeline for one dataset. Returns summary list."""
    results_dir = workdir / "cache" / name
    results_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    run_status, request_times = run_lirical_all(entries, results_dir, args)
    summary = collect_results(entries, results_dir, run_status, request_times)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workdir = Path(script_dir) / "lirical_benchmarks"
    workdir.mkdir(parents=True, exist_ok=True)

    all_cases = load_all_datasets(Path(args.data_dir), args.datasets)

    all_summaries = {}
    for dataset_name, cases in all_cases.items():
        print(f"Processing dataset: {dataset_name} with {len(cases)} cases")
        summary = run_dataset(dataset_name, cases, args, workdir)
        all_summaries[dataset_name] = summary
        save_summary_tsv(summary, workdir / f"{dataset_name}_summary.tsv")

    print("Generating final summary statistics")
    for dataset_name, summary in all_summaries.items():
        out_path = workdir / f"{dataset_name}_stats.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            print_stats(
                dataset_name, compute_stats(summary), f, reference=PAPER_LIRICAL_PUBLIC
            )
        print(f"  Stats written to {out_path}")


if __name__ == "__main__":
    main()
