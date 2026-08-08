# Comparing Methods

`compare_methods.py` aggregates the standardized `<dataset>_summary.tsv` files
produced by every `run_<tool>.py` runner and compares Top-1/5/10 accuracy (and
more) across all tools for a given dataset.


## 1. What It Does

<!-- % new "Implementation" section describing the script -->
It auto-discovers every `output/validation_tools/*_benchmarks/<dataset>_summary.tsv`
file, groups them by dataset, and, for each dataset, produces three sections:

1. **Recall@k and MRR** — Recall@1/5/10 (configurable), Mean Reciprocal Rank,
   the found/total count, and average query time per method. Methods without
   timing data (e.g. local PhenoBrain) show `/` in that column.
2. **Agreement analysis** — consensus cases (all methods found it), hard cases
   (no method found it), easy cases (at least one method ranked it #1), and
   unique finds (only one method found it), with simple bar-chart distributions.
3. **Per-case rank matrix** — one row per case, one column per method, showing
   the rank each method assigned to the correct disease (`-` = outside `--max-rank`).

The method name shown in every table is derived from the benchmark folder name
(`lirical_benchmarks` → `lirical`, `phenobrain_local_benchmarks` →
`phenobrain_local`, etc.), so any new runner is picked up automatically.


## 2. Running It

No file paths are required. Discovery is automatic. The simplest invocation
compares **every** dataset it can find:

```bash
# Compare every dataset across all benchmark folders
python3 compare_methods.py

# Compare a single dataset
python3 compare_methods.py --dataset mme

# List available datasets and which methods have results for each
python3 compare_methods.py --list-datasets

# Write a single dataset's report to a custom file (use '-' for stdout)
python3 compare_methods.py --dataset mme --output results/mme_comparison.txt
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset to compare (e.g. `mme`). Omit to compare all discovered datasets. |
| `--list-datasets` | List datasets and their available methods, then exit. |
| `--topk` | Top-k values for the Recall table. Default: `1 5 10`. |
| `--max-rank` | Max rank shown in the agreement analysis and rank matrix. Default: `10`. |
| `--output` | Output file. Default: `results/<dataset>_comparison.txt`. Use `-` for stdout. |

## 3. Output

Reports are written to `output/validation_tools/results/<dataset>_comparison.txt`
(one per dataset).

If a method is missing some cases that other methods have, a `[WARN]` line is
printed listing the missing case IDs, usually a sign of an interrupted run or a
dataset mismatch between tools.