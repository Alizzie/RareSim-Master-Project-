#!/usr/bin/env python3
"""
compare_methods.py — Compare multiple method benchmark summaries.

Reads one TSV summary per method (output of run_lirical.py or other runners)
and produces:
  1. Recall@k and MRR table per method
  2. Agreement analysis (consensus / hard / easy cases, unique finds)
  3. Per-case rank matrix

Usage:
    python3 compare_methods.py \
        --summaries lirical=lirical_bench/MME_summary.tsv \
                    phen2gene=phen2gene_bench/MME_summary.tsv \
        --topk 1 5 10 \
        --output results.txt
"""

import argparse
import sys
from pathlib import Path
from typing import TextIO
from utils import compute_stats, compute_mrr, load_summary_tsv


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Compare method benchmark summaries")
    p.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Method summaries as name=path pairs, e.g. lirical=bench/MME_summary.tsv",
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
        help="Write output to this file (default: stdout)",
    )
    return p.parse_args()


# ── Loader ────────────────────────────────────────────────────────────────────
def load_summaries(summary_args: list[str]) -> dict[str, list[dict]]:
    summaries = {}
    for arg in summary_args:
        if "=" not in arg:
            raise ValueError(f"Expected name=path format, got: {arg}")
        name, path = arg.split("=", 1)
        summaries[name] = load_summary_tsv(Path(path))
    return summaries


# ── Align cases across methods ────────────────────────────────────────────────
def align_cases(summaries: dict[str, list[dict]]) -> tuple[list[str], dict]:
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
    col_w = 35
    k_w = 7
    header = f"  {'Method':<{col_w}}" + "".join(f"{'R@'+str(k):>{k_w}}" for k in top_ks)
    header += f"{'MRR':>{k_w}}   Found"
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
        line += f"{mrr:>{k_w}.4f}  {found}/{n}"
        f.write(f"{line}\n")

    f.write(f"{sep}\n")


# ── Section 2: Agreement analysis ────────────────────────────────────────────
def print_agreement(case_ids, aligned, methods, max_rank, f: TextIO):
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
def print_rank_matrix(case_ids, aligned, methods, max_rank, f: TextIO):
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    summaries = load_summaries(args.summaries)
    methods = list(summaries.keys())
    case_ids, aligned = align_cases(summaries)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        f = open(args.output, "w", encoding="utf-8")
    else:
        f = sys.stdout

    try:
        print_recall_table(summaries, args.topk, f)
        print_agreement(case_ids, aligned, methods, args.max_rank, f)
        print_rank_matrix(case_ids, aligned, methods, args.max_rank, f)
    finally:
        if f is not sys.stdout:
            f.close()


if __name__ == "__main__":
    main()
