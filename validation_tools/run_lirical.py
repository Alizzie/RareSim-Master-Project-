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
  --data-dir validation_tools/datasets/PhenoBrainBenchmarkDatasets \
  --lirical-jar "/Users/eli/Documents/Uni/Master Project/Tools/LIRICAL/lirical-cli/target/lirical-cli-2.4.0.jar" \
  --lirical-data ~/lirical-data \
  --workdir validation_tools/lirical_bench \
  --skip-existing --datasets "HMS"
"""

import subprocess
import argparse
from pathlib import Path

from utils import (
    DATASET_NAMES,
    PUBLIC_TEST_DATASETS,
    load_all_datasets,
    parse_lirical_tsv,
    find_best_rank,
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
        "--data-dir", required=True, help="Directory containing the 6 JSON files"
    )
    p.add_argument("--lirical-jar", required=True, help="Path to LIRICAL JAR")
    p.add_argument(
        "--lirical-data", required=True, help="Path to LIRICAL data directory"
    )
    p.add_argument("--workdir", default="lirical_bench", help="Output directory")
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
    results_dir.mkdir(parents=True, exist_ok=True)
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


def collect_results(entries, results_dir, run_status):
    """Phase 2: parse TSV outputs and build summary list."""
    summary = []
    for case_id, hpo_ids, confirmed_diseases in entries:
        tsv_path = results_dir / f"{case_id}.tsv"
        lirical_results = parse_lirical_tsv(tsv_path)
        rank, matched_id = find_best_rank(lirical_results, confirmed_diseases)
        summary.append(
            {
                "case_id": case_id,
                "n_hpo": len(hpo_ids),
                "confirmed_diseases": ";".join(confirmed_diseases),
                "rank": rank,
                "matched_id": matched_id,
                "total_candidates": len(lirical_results),
                "lirical_ok": run_status.get(case_id, False),
            }
        )

        print(
            f"  {case_id}: rank={rank}, matched_id={matched_id}, "
            f"n_hpo={len(hpo_ids)}, candidates={len(lirical_results)}, "
            f"lirical_ok={run_status.get(case_id, False)}"
        )
    return summary


def run_dataset(name, cases, args, workdir):
    """Run the full LIRICAL pipeline for one dataset. Returns summary list."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({len(cases)} cases)")

    results_dir = workdir / name / "results"

    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    run_status = run_lirical_all(entries, results_dir, args)
    summary = collect_results(entries, results_dir, run_status)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Load all requested datasets
    all_cases = load_all_datasets(Path(args.data_dir), args.datasets)

    # Run LIRICAL and collect summaries
    all_summaries = {}
    for name, cases in all_cases.items():
        summary = run_dataset(name, cases, args, workdir)
        all_summaries[name] = summary
        save_summary_tsv(summary, workdir / f"{name}_summary.tsv")

    # ── Per-dataset report ────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("RESULTS PER DATASET")
    print(f"{'='*60}")
    for name, summary in all_summaries.items():
        print_stats(name, compute_stats(summary))

    # ── Combined public test set (MME + HMS + LIRICAL) vs. paper ─────────────
    public_combined = []
    for name in PUBLIC_TEST_DATASETS:
        if name in all_summaries:
            public_combined.extend(all_summaries[name])

    if public_combined:
        print(f"\n{'='*60}")
        print("PUBLIC TEST SET (MME + HMS + LIRICAL) vs. paper")
        print(f"{'='*60}")
        pub_stats = compute_stats(public_combined)
        print_stats("Combined public", pub_stats, reference=PAPER_LIRICAL_PUBLIC)
        save_summary_tsv(public_combined, workdir / "PUBLIC_COMBINED_summary.tsv")

    print(f"\nAll outputs written to: {workdir}/")


if __name__ == "__main__":
    main()
