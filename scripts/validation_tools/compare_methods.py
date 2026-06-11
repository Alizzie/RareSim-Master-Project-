"""
compare_methods.py — Compare multiple method benchmark summaries.

Reads one TSV summary per method (output of run_<tool>.py runners) and produces:
  1. Recall@k and MRR table per method
  2. Agreement analysis (consensus / hard / easy cases, unique finds)
  3. Per-case rank matrix

Benchmark folders are auto-discovered using the pattern *_benchmarks/.
Available datasets are inferred from the summary TSV files found in those folders.

Usage:
  # Compare all methods for a specific dataset
  python3 compare_methods.py --dataset mme

  # Compare a specific dataset and write to a custom output file
  python3 compare_methods.py --dataset mme --output results/mme_comparison.txt

  # List all available datasets across benchmark folders
  python3 compare_methods.py --list-datasets
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import TextIO
from _utils import compute_stats, compute_mrr, load_summary_tsv
from raresim.utils.paths import OUTPUTS_DIR

VAL_OUTPUTS_DIR = OUTPUTS_DIR / "validation_tools"
DEFAULT_OUTPUT_DIR = VAL_OUTPUTS_DIR / "results"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare method benchmark summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name to compare (e.g. 'mme'). "
            "If omitted, all datasets found across benchmark folders are compared."
        ),
    )
    p.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets across benchmark folders and exit.",
    )
    p.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Top-k values to report (default: 1 5 10)",
    )
    p.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="Max rank shown in rank matrix (default: 10)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write output to this file. "
            "Defaults to results/<dataset>_comparison.txt. "
            "Use '-' for stdout."
        ),
    )
    return p.parse_args()


# ── Dataset discovery ─────────────────────────────────────────────────────────
def discover_available_datasets() -> dict[str, list[Path]]:
    """
    Scan all *_benchmarks/ folders and collect available dataset summary files.

    Returns:
        Dict mapping dataset name -> list of summary TSV Paths across all methods.
    """
    pattern = str(VAL_OUTPUTS_DIR / "*_benchmarks" / "*_summary.tsv")
    datasets: dict[str, list[Path]] = {}
    for path_str in glob.glob(pattern):
        path = Path(path_str)
        dataset_name = path.stem.replace("_summary", "")
        datasets.setdefault(dataset_name, []).append(path)
    return datasets


def list_datasets() -> None:
    """Print all available datasets and which methods have results for them."""
    available = discover_available_datasets()
    if not available:
        print("No benchmark summaries found. Run at least one tool first.")
        return
    print(f"Available datasets ({len(available)} found):\n")
    for name, paths in sorted(available.items()):
        methods = [p.parent.name.replace("_benchmarks", "") for p in paths]
        print(f"  {name:<20} methods: {', '.join(sorted(methods))}")


# ── Loader ────────────────────────────────────────────────────────────────────
def load_summaries(dataset: str) -> dict[str, list[dict]]:
    """
    Find and load all *_benchmarks/<dataset>_summary.tsv files.

    Returns:
        Dict mapping method name -> list of case result dicts.

    Raises:
        FileNotFoundError if no summary files are found for the dataset.
    """
    pattern = str(VAL_OUTPUTS_DIR / "*_benchmarks" / f"{dataset}_summary.tsv")
    summaries = {}
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        method = path.parent.name.replace("_benchmarks", "")
        summaries[method] = load_summary_tsv(path)
        print(f"  Loaded '{method}': {len(summaries[method])} cases from {path}")

    if not summaries:
        raise FileNotFoundError(
            f"No summary files found for dataset '{dataset}'.\n"
            f"  Looked for: {pattern}\n"
            f"  Run --list-datasets to see what is available."
        )
    return summaries


# ── Align cases across methods ────────────────────────────────────────────────
def align_cases(summaries: dict[str, list[dict]]) -> tuple[list[str], dict]:
    """
    Align cases across all methods by case_id.

    Warns about cases that appear in some methods but not others, which can
    indicate mismatched runs or missing results.

    Returns:
        (sorted case_ids, aligned dict mapping case_id -> {gt, ranks})
    """
    method_case_ids: dict[str, set[str]] = {
        method: {row["case_id"] for row in rows} for method, rows in summaries.items()
    }

    all_case_ids = set().union(*method_case_ids.values())
    for method, ids in method_case_ids.items():
        missing = all_case_ids - ids
        if missing:
            print(
                f"  [WARN] '{method}' is missing {len(missing)} case(s) "
                f"present in other methods: {sorted(missing)[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

    aligned = {}
    for method, rows in summaries.items():
        for row in rows:
            cid = row["case_id"]
            if cid not in aligned:
                gt_raw = row.get("confirmed_diseases", "")
                aligned[cid] = {
                    "gt": [d.strip() for d in gt_raw.split(";") if d.strip()],
                    "ranks": {},
                }
            r = row.get("rank")
            aligned[cid]["ranks"][method] = int(r) if r and r != "None" else None

    case_ids = sorted(aligned.keys())
    return case_ids, aligned


# ── Section 1: Recall@k and MRR ──────────────────────────────────────────────
def print_recall_table(
    summaries: dict[str, list[dict]], top_ks: list[int], f: TextIO
) -> None:
    """Print a table comparing Recall@k and MRR for each method."""
    col_w = 35
    k_w = 7
    header = f"  {'Method':<{col_w}}" + "".join(f"{'R@'+str(k):>{k_w}}" for k in top_ks)
    header += f"{'MRR':>{k_w}}\tFound\tAvg. Query Time (s)"
    sep = "=" * (len(header) + 2)

    f.write(f"\n{sep}\n")
    f.write("  Method Comparison — Recall@k and MRR\n")
    f.write(f"{sep}\n")
    f.write(f"{header}\n")
    f.write(f"  {'-' * (len(header) - 2)}\n")

    for method, rows in summaries.items():
        stats = compute_stats(rows, top_ks)
        mrr = compute_mrr(rows)
        n, found = stats["n"], stats["found"]
        line = f"  {method:<{col_w}}"
        for k in top_ks:
            line += f"{stats['topk'][k]:>{k_w}.4f}"
        line += f"{mrr:>{k_w}.4f}\t{found}/{n}\t{stats['mean_query_time']}"
        f.write(f"{line}\n")

    f.write(f"{sep}\n")


# ── Section 2: Agreement analysis ────────────────────────────────────────────
def print_agreement(
    case_ids: list[str], aligned: dict, methods: list[str], max_rank: int, f: TextIO
) -> None:
    """Analyze agreement between methods on which cases were found and at what ranks."""
    n = len(case_ids)
    consensus, hard, easy, unique_finds = [], [], [], []
    method_found_counts = {cid: 0 for cid in case_ids}

    for cid in case_ids:
        ranks = aligned[cid]["ranks"]
        found_by = [
            m for m in methods if ranks.get(m) is not None and ranks[m] <= max_rank
        ]
        method_found_counts[cid] = len(found_by)

        if len(found_by) == len(methods):
            consensus.append(cid)
        if len(found_by) == 0:
            hard.append(cid)
        if any(ranks.get(m) == 1 for m in methods):
            easy.append(cid)
        if len(found_by) == 1:
            unique_finds.append((cid, found_by[0]))

    sep = "=" * 72
    f.write(f"\n{sep}\n")
    f.write("  Method Agreement Analysis\n")
    f.write(f"{sep}\n")
    f.write(f"  Total cases : {n}\n")
    f.write(f"  Consensus   : {len(consensus)} cases — all methods found it\n")
    f.write(f"  Hard cases  : {len(hard)} cases — no method found it\n")
    f.write(f"  Easy cases  : {len(easy)} cases — at least one method ranked it #1\n")
    f.write(f"  Unique finds: {len(unique_finds)} cases — only one method found it\n")

    f.write(f"\n  How many methods found the correct disease per case:\n")
    max_count = len(methods)
    dist = {i: 0 for i in range(max_count + 1)}
    for cid in case_ids:
        dist[method_found_counts[cid]] += 1
    bar_scale = 40 / max(dist.values()) if dist.values() else 1
    for i in range(max_count + 1):
        bar = "█" * int(dist[i] * bar_scale)
        f.write(f"    {i} methods: {dist[i]:>3} cases  {bar}\n")

    f.write(f"\n  How many methods found correct disease at each rank:\n")
    rank_dist = {}
    for cid in case_ids:
        for m in methods:
            r = aligned[cid]["ranks"].get(m)
            if r is not None and r <= max_rank:
                rank_dist[r] = rank_dist.get(r, 0) + 1
    if rank_dist:
        bar_scale = 40 / max(rank_dist.values())
        for r in sorted(rank_dist):
            bar = "█" * int(rank_dist[r] * bar_scale)
            f.write(f"    rank_{r}: {rank_dist[r]:>2}  {bar}\n")

    if unique_finds:
        f.write(f"\n  Unique finds (only one method found it):\n")
        for cid, method in unique_finds:
            r = aligned[cid]["ranks"][method]
            gt = aligned[cid]["gt"]
            f.write(f"    {cid} | gt={gt} | {method} @ rank {r}\n")

    if hard:
        f.write(f"\n  Hard cases (no method found correct disease):\n")
        for cid in hard[:10]:
            f.write(f"    {cid}\n")
        if len(hard) > 10:
            f.write(f"    ... and {len(hard) - 10} more\n")

    f.write(f"{sep}\n")


# ── Section 3: Per-case rank matrix ──────────────────────────────────────────
def print_rank_matrix(
    case_ids: list[str], aligned: dict, methods: list[str], max_rank: int, f: TextIO
) -> None:
    """Print a matrix of ranks per case and method, with cases as rows and methods as columns."""
    short = {m: m[:9] for m in methods}
    col_w = 12
    gt_w = 20

    sep = "=" * 72
    f.write(f"\n{sep}\n")
    f.write(f"  Per-Case Rank Matrix  (- = not found in top {max_rank})\n")
    f.write(f"{sep}\n")

    hdr = f"  {'Case':<{col_w}} {'GT':<{gt_w}}"
    for m in methods:
        hdr += f"{short[m]:>{col_w}}"
    f.write(f"{hdr}\n")
    f.write(f"  {'-' * (len(hdr) - 2)}\n")

    for cid in case_ids:
        gt_str = ",".join(aligned[cid]["gt"])[: gt_w - 1]
        row = f"  {cid:<{col_w}} {gt_str:<{gt_w}}"
        for m in methods:
            r = aligned[cid]["ranks"].get(m)
            cell = str(r) if (r is not None and r <= max_rank) else "-"
            row += f"{cell:>{col_w}}"
        f.write(f"{row}\n")

    f.write(f"{sep}\n")


# ── Per-dataset runner ────────────────────────────────────────────────────────
def run_comparison(dataset: str, args) -> None:
    """Load summaries and run all three comparison sections for one dataset."""
    print(f"\nComparing methods for dataset: {dataset}")
    summaries = load_summaries(dataset)
    methods = list(summaries.keys())
    case_ids, aligned = align_cases(summaries)

    # Resolve output path
    if args.output and str(args.output) == "-":
        f = sys.stdout
        should_close = False
    else:
        out_path = args.output or (DEFAULT_OUTPUT_DIR / f"{dataset}_comparison.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(out_path, "w", encoding="utf-8")
        should_close = True
        print(f"  Writing results to {out_path}")

    try:
        print_recall_table(summaries, args.topk, f)
        print_agreement(case_ids, aligned, methods, args.max_rank, f)
        print_rank_matrix(case_ids, aligned, methods, args.max_rank, f)
    finally:
        if should_close:
            f.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.list_datasets:
        list_datasets()
        return

    if args.dataset:
        run_comparison(args.dataset, args)
    else:
        available = discover_available_datasets()
        if not available:
            print("No benchmark summaries found. Run at least one tool first.")
            return
        print(
            f"No --dataset specified. Running comparison for all {len(available)} found:"
        )
        for name in sorted(available):
            print(f"  - {name}")
        for name in sorted(available):
            run_comparison(name, args)


if __name__ == "__main__":
    main()
