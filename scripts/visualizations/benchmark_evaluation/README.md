# RareSim Evaluation Visualization

Builds the figures, metric tables, and a single self-contained HTML report that
compare RareSim methods against external validation tools on the PhenoBrain
benchmark datasets (HMS, MME, LIRICAL, RAMEDIS, PUMCH_L, PUMCH-ADM).

It answers seven questions:

1. **Q1** — Which method performs best? (Recall@10 heatmap + per-dataset tables)
2. **Q2** — Which method ranks the correct disease highest? (Recall@k curves)
3. **Q3** — Which method is too slow for its performance? (speed vs Recall@10)
4. **Q4** — Are some datasets much harder? (best vs mean Recall@10)
5. **Q5** — Do validation tools beat RareSim methods? (best per system type + gap)
6. **Q6** — Does combining methods (RRF) beat the best single method?
7. **Q7** — How do the method families compare overall?

---

## Expected layout

The loader discovers everything by globbing, so folder layout matters more than
exact filenames. RareSim results are read from `*evaluation*.json` (preferred) and/or `*summary*.tsv`.
Validation-tool results are read from `*summary*.tsv`.

```text
RareSim-Master-Project-/
├── outputs/
│   ├── evaluation/
│   │   ├── HMS/HMS_evaluation.json          # or HMS_summary.tsv
│   │   ├── MME/MME_evaluation.json
│   │   ├── LIRICAL/…
│   │   ├── RAMEDIS/…
│   │   ├── PUMCH_L/…
│   │   └── PUMCH-ADM/…
│   └── validation_tools/
│       ├── phenobrain_benchmarks/hms_summary.tsv
│       ├── phenobrain_benchmarks/mme_summary.tsv
│       ├── dx29_benchmarks/…
│       └── phenomizer_benchmarks/…
│
└── scripts/visualizations/benchmark_evaluation/
    ├── config.py
    ├── load_results.py
    ├── plot_evaluation_questions.py
    ├── make_evaluation_report.py
    └── requirements.txt
```

### Validation-tool path convention (drives Q5)

```text
validation_tools/<tool>_benchmarks/<dataset>_summary.tsv
```

`validation_tools/phenobrain_benchmarks/hms_summary.tsv` →
tool = **PhenoBrain**, dataset = **HMS**, system_type = **Validation tool**.
The parent folder names the tool; the filename names the dataset.

### Required TSV / JSON fields

- **Validation TSV**: must have a `rank` column. `case_id`, `method`,
  `query_time_sec` are recommended (missing ranks → disease not found).
- **RareSim JSON**: `n_cases`, `method_metrics` (per method: `recall_1/3/5/10/20`,
  `mrr`, `ndcg`, `median_rank`, `found`), `method_avg_seconds`, and optionally
  `rank_matrix` (used for Q4 case-agreement).

---

## Run

From the **project root** (the `-m` form requires it, so the package imports
resolve):

# 1. generate figures + CSV tables
python -m scripts.visualizations.benchmark_evaluation.plot_evaluation_questions \
  --raresim outputs/evaluation \
  --validation outputs/validation_tools \
  --output outputs/evaluation_visual_questions

```

RareSim only (skip the tool comparison — Q5/Q6 simply omit the tool bars):

```bash
python -m scripts.visualizations.benchmark_evaluation.plot_evaluation_questions \
  --raresim outputs/evaluation \
  --output outputs/evaluation_visual_questions
```
`--top-n` controls how many methods appear in the Q2 curves (default 7).

# 2. build the self-contained HTML report from those figures
python -m scripts.visualizations.benchmark_evaluation.make_evaluation_report \
  --plots outputs/evaluation_visual_questions \
  --output outputs/evaluation_visual_questions/evaluation_report.html

---

## Outputs

```text
outputs/evaluation_visual_questions/
├── evaluation_report.html                         # the report (open this)
├── combined_metrics.csv                           # every metric, every method
├── q1_best_method_recall10_heatmap.png
├── q1_best_methods_by_dataset.csv
├── q2_<dataset>_recall_curve_top_methods.png
├── q3_<dataset>_speed_vs_recall10.png
├── q4_dataset_difficulty.png
├── q4_dataset_difficulty_summary.csv
├── q5_validation_vs_raresim_best_recall10.png
├── q5_validation_minus_raresim_difference.png
├── q5_best_by_system_type.csv
├── q5_difference_table.csv
├── q6_ensemble_vs_single.png
├── q6_ensemble_gain.csv
├── q7_family_overview.png
├── q7_family_overview.csv
└── case_agreement_by_case.csv                      # only if rank_matrix present
```

---

## Adding a method, tool, or dataset

Everything is discovered from disk. What auto-works vs. what needs a config edit:

| You add a… | Required | Optional |
|---|---|---|
| **RareSim method** (new key in `method_metrics`) | nothing | `METHOD_LABELS[key]` for a readable name; a `method_family()` rule **only if** it's a new family, else it shows as *Other RareSim method* |
| **Validation tool** (new `<tool>_benchmarks/` folder) | nothing | `VALIDATION_TOOL_LABELS[folder]` for correct casing, else it's title-cased from the folder name |
| **Dataset** (new `evaluation/FOO/…`) | **add `"FOO"` to `DATASETS`** in `config.py`, or it is filtered out | `DATASET_COLORS["FOO"]` (else grey); a `DATASET_NAME_MAP` entry if the folder/file token isn't already the canonical name |

Why a dataset needs the extra line: `DATASETS` is an **allow-list**.
The loader keeps only those datasets and uses the same list for display order.
Anything else found on disk is ignored — and the run prints
`[load_results] ignoring datasets outside the report set: [...]`, so check that
line if a new dataset doesn't appear.

`config.py` is the single place to edit. Method/tool/dataset names map through
`METHOD_LABELS`, `VALIDATION_TOOL_LABELS`, and `DATASET_NAME_MAP` (keys are
**lowercased** folder/file tokens). The table columns come from
`SUMMARY_METRIC_COLUMNS`; colours from `SYSTEM_TYPE_COLORS` / `DATASET_COLORS`.

---
