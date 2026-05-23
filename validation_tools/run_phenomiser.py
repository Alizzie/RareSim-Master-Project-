#!/usr/bin/env python3
"""
run_phenomizer.py — Run Phenomizer on PhenoBrain benchmark datasets.

Usage:
    python3 validation_tools/run_phenomiser.py \
        --data-dir validation_tools/datasets/PhenoBrainBenchmarkDatasets \
        --phenomizer-jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar \
        --hp-obo ~/phenomiser_data/hp.obo \
        --hpoa ~/phenomiser_data/phenotype.hpoa --datasets MME
"""

import os
import subprocess
import argparse
import csv
import time
from pathlib import Path

from utils import (
    DATASET_NAMES,
    load_all_datasets,
    save_summary_tsv,
    compute_stats,
    print_stats,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run Phenomizer on PhenoBrain benchmark datasets"
    )
    p.add_argument(
        "--data-dir",
        default="validation_tools/datasets/PhenoBrainBenchmarkDatasets",
        help="Directory containing the JSON dataset files",
    )
    p.add_argument("--phenomizer-jar", required=True, help="Path to Phenomizer JAR")
    p.add_argument("--hp-obo", required=True, help="Path to hp.obo ontology file")
    p.add_argument(
        "--hpoa", required=True, help="Path to phenotype.hpoa annotations file"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="Datasets to run (default: all)",
    )
    p.add_argument("--java", default="java", help="Path to java executable")
    p.add_argument("--xmx", default="32g", help="Java heap size (default: 32g)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip case if output already exists (resume mode)",
    )
    return p.parse_args()


# ── Phenomizer runner ─────────────────────────────────────────────────────────
def run_phenomizer_case(
    java: str,
    xmx: str,
    jar: str,
    hp_obo: str,
    hpoa: str,
    case_id: str,
    hpo_ids: list[str],
    out_dir: Path,
    skip_existing: bool,
) -> bool:
    """Run Phenomizer for a single case. Returns True on success."""
    out_file = out_dir / f"{case_id}.txt"
    if skip_existing and out_file.exists():
        return True

    cmd = [
        java,
        f"-Xmx{xmx}",
        "-jar",
        jar,
        "query",
        "-hpo",
        hp_obo,
        "-da",
        hpoa,
        "-query",
        ",".join(hpo_ids),
        "-o",
        str(out_file),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [WARN] Phenomizer failed for {case_id}: {r.stderr[-200:]}")
        return False
    return True


# ── Per-dataset pipeline ──────────────────────────────────────────────────────
def run_phenomizer_all(entries, results_dir, args) -> dict[str, bool]:
    """Phase 1: run Phenomizer on every case. Returns dict case_id -> bool."""
    status = {}
    request_times = {}
    for i, (case_id, hpo_ids, _) in enumerate(entries):
        start_time = time.time()
        ok = run_phenomizer_case(
            java=args.java,
            xmx=args.xmx,
            jar=args.phenomizer_jar,
            hp_obo=args.hp_obo,
            hpoa=args.hpoa,
            case_id=case_id,
            hpo_ids=hpo_ids,
            out_dir=results_dir,
            skip_existing=args.skip_existing,
        )
        status[case_id] = ok
        request_times[case_id] = time.time() - start_time
        if (i + 1) % 50 == 0:
            print(f"  Phenomizer: {i+1}/{len(entries)} done")
    return status, request_times


def parse_phenomizer_output(out_path: Path) -> list[dict]:
    """
    Parse a Phenomizer output file.

    Columns: diseaseId  diseaseName  p  adjust_p  similarityScore
    Rows are kept in file order (Phenomizer sorts by adjust_p asc,
    then similarityScore desc), and ranks are assigned accordingly.
    Returns empty list if file does not exist or is empty.
    """
    if not out_path.exists():
        return []

    lines = [
        l
        for l in out_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    if not lines:
        return []

    reader = csv.DictReader(lines, delimiter="\t")
    rows = []
    for rank, row in enumerate(reader, start=1):
        rows.append(
            {
                "rank": rank,
                "disease_id": row.get("diseaseId", "").strip(),
                "disease_name": row.get("diseaseName", "").strip(),
                "p": row.get("p", "").strip(),
                "adjust_p": row.get("adjust_p", "").strip(),
                "score": row.get("similarityScore", "").strip(),
            }
        )
    return rows


def find_best_rank(
    results: list[dict], confirmed_diseases: list[str]
) -> tuple[int | None, str | None, str | None]:
    """Find the best (lowest) rank among all confirmed disease IDs."""
    confirmed = {d.upper() for d in confirmed_diseases}
    for row in results:  # already in ranked order
        if row["disease_id"].upper() in confirmed:
            return row["rank"], row["disease_id"], row["score"]
    return None, None, None


def collect_results(entries, results_dir, run_status, request_times) -> list[dict]:
    """Phase 2: parse outputs and build summary list."""
    summary = []
    for case_id, hpo_ids, confirmed_diseases in entries:
        out_path = results_dir / f"{case_id}.txt"
        results = parse_phenomizer_output(out_path)
        rank, matched_id, score = find_best_rank(results, confirmed_diseases)
        summary.append(
            {
                "case_id": case_id,
                "n_hpo": len(hpo_ids),
                "confirmed_diseases": ";".join(confirmed_diseases),
                "rank": rank,
                "matched_id": matched_id,
                "score": score,
                "status": run_status.get(case_id, False),
                "query_time_sec": request_times.get(case_id, None),
            }
        )
        print(
            f"  {case_id}: rank={rank}, matched_id={matched_id}, "
            f"n_hpo={len(hpo_ids)}, score={score}, "
            f"status={run_status.get(case_id, False)}, query_time_sec={request_times.get(case_id, None)}"
        )
    return summary


def run_dataset(name, cases, args, workdir) -> list[dict]:
    """Run the full Phenomizer pipeline for one dataset. Returns summary list."""
    results_dir = Path(workdir) / "cache" / name
    results_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    run_status, request_times = run_phenomizer_all(entries, results_dir, args)
    summary = collect_results(entries, results_dir, run_status, request_times)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    workdir = script_dir / "phenomizer_benchmarks"
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
            print_stats(dataset_name, compute_stats(summary), f)
        print(f"  Stats written to {out_path}")


if __name__ == "__main__":
    main()
