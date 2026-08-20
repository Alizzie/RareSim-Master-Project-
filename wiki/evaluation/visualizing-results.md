# Visualizing Evaluation Results

## Purpose

`evaluator.py` produces per-dataset JSON/TXT/TSV output (see [evaluator-and-metrics.md](evaluator-and-metrics.md)). This toolkit is a separate layer on top of that: it reads results across *all* your datasets (and, optionally, external validation-tool results), and turns them into comparison figures, CSV tables, and a single self-contained HTML report.

File location:

```text
scripts/evaluation/benchmark_visualization/
    config.py
    load_results.py
    plot_evaluation_questions.py
    make_evaluation_report.py
    README.md
```

It answers seven fixed questions, each producing its own figure(s) and a companion CSV:

```text
Q1  Which method performs best?          Recall@10 heatmap (method x dataset) + best-per-dataset table
Q2  Which method ranks the disease highest?  Recall@k curves per dataset, top-N methods
Q3  Which method is too slow for its performance?  Runtime vs Recall@10 scatter per dataset
Q4  Are some datasets much harder?       Best vs mean Recall@10 per dataset, + hard-case rate
Q5  Do validation tools beat RareSim?    Best-per-system-type bars + validation-minus-RareSim gap
Q6  Does combining methods (RRF) help?   Best ensemble vs best single method per dataset
Q7  How do method families compare?      Mean Recall@10 by family, across all datasets
```


## Where it fits in the workflow

```text
run_*.py batch runners  ->  outputs/evaluation/<DATASET>/cache/case_*.json
                                        |
                                  evaluator.py
                                        |
              outputs/evaluation/<DATASET>/<DATASET>_evaluation.json (+ _summary.tsv, etc.)
                                        |
                     plot_evaluation_questions.py   (this toolkit, step 1)
                                        |
                        outputs/evaluation_visual_questions/*.png, *.csv
                                        |
                       make_evaluation_report.py   (this toolkit, step 2)
                                        |
                    outputs/evaluation_visual_questions/evaluation_report.html
```

Run it after you've already run the evaluator for every dataset you want to compare — it doesn't call any batch runner or the evaluator itself, it only reads finished output files from disk.


## Expected input layout

The loader discovers files by globbing, so folder layout matters more than exact filenames.

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
└── scripts/evaluation/benchmark_visualization/
```

RareSim results are read from `*evaluation*.json` (preferred) and/or `*summary*.tsv`, searched recursively under `--raresim`. Validation-tool results are read from `*summary*.tsv` under `--validation`.

### Validation-tool path convention (drives Q5)

```text
validation_tools/<tool>_benchmarks/<dataset>_summary.tsv
```

`validation_tools/phenobrain_benchmarks/hms_summary.tsv` → tool = **PhenoBrain**, dataset = **HMS**, system type = **Validation tool**. The parent folder names the tool; the filename names the dataset. For how those TSVs get generated in the first place, see the Validation section of the wiki (e.g. `../validation/tools-overview.md`, `../validation/phenobrain.md`).

### Required fields

```text
Validation TSV   must have a "rank" column.
                 "case_id", "method", "query_time_sec" are recommended
                 (rows with no rank are treated as "not found").

RareSim JSON     "n_cases", "method_metrics" (per method:
                 recall_1/3/5/10/20, mrr, ndcg, median_rank, found),
                 "method_avg_seconds", and optionally "rank_matrix"
                 (used for Q4's hard-case rate).
```

These are exactly the fields `evaluator.py` writes into `<DATASET>_evaluation.json` — see [evaluator-and-metrics.md](evaluator-and-metrics.md#output-files).


## Running it

Two steps, from the **project root** (the `-m` form requires it, so the package imports resolve):

```bash
# 1. generate figures + CSV tables
python -m scripts.evaluation.benchmark_visualization.plot_evaluation_questions \
    --raresim outputs/evaluation \
    --validation outputs/validation_tools \
    --output outputs/evaluation_visual_questions

# 2. build the self-contained HTML report from those figures
python -m scripts.evaluation.benchmark_visualization.make_evaluation_report \
    --plots outputs/evaluation_visual_questions \
    --output outputs/evaluation_visual_questions/evaluation_report.html
```

RareSim only, skipping the validation-tool comparison (Q5/Q6 simply omit the tool bars):

```bash
python -m scripts.evaluation.benchmark_visualization.plot_evaluation_questions \
    --raresim outputs/evaluation \
    --output outputs/evaluation_visual_questions
```

`--top-n` controls how many methods appear in the Q2 recall curves (default 7).


## Outputs

```text
outputs/evaluation_visual_questions/
├── evaluation_report.html                         # open this — the full report
├── combined_metrics.csv                           # every metric, every method, every dataset
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
└── case_agreement_by_case.csv                      # only if rank_matrix was present in the source JSON
```

`evaluation_report.html` embeds every PNG as a base64 data URI and inlines its own CSS, so it's a single file you can open directly or send to someone without the rest of the `outputs/` tree.


## Name resolution (methods, tools, datasets)

`load_results.py` normalizes raw identifiers into display-friendly labels and groupings:

```python
normalize_dataset_name(value)   # folder/file token -> canonical dataset name (via DATASET_NAME_MAP)
clean_method_label(method)      # raw method key -> readable label (via METHOD_LABELS)
validation_tool_label(path)     # "<tool>_benchmarks" parent folder -> tool label (via VALIDATION_TOOL_LABELS)
method_family(method)           # raw method key -> family bucket, e.g. "Semantic", "Set-based", "Ensemble"
system_type_for_raresim(method) # "RareSim method" vs "Ensemble" (anything named ensemble_*)
```

`method_family()` classifies by name pattern: `ensemble_*` → Ensemble, `semantic_*` → Semantic, `set_*` → Set-based, `tfidf`/`tfidf_cosine` → TF-IDF, `hpo2vec` → HPO2Vec, anything with "mistral" or "llm" in the name → LLM, anything with "autoencoder" → Autoencoder, anything with "bert"/"minilm"/"transformer" → Transformer encoder, and everything else falls into "Other RareSim method". A method not covered by any rule still shows up in every chart.


## Adding a method, tool, or dataset

Everything is discovered from disk. What auto-works vs. what needs a `config.py` edit:

| You add a… | Required | Optional |
|---|---|---|
| **RareSim method** (new key in `method_metrics`) | nothing | `METHOD_LABELS[key]` for a readable name; a `method_family()` rule only if it's a genuinely new family, else it shows as *Other RareSim method* |
| **Validation tool** (new `<tool>_benchmarks/` folder) | nothing | `VALIDATION_TOOL_LABELS[folder]` for correct casing, else it's title-cased from the folder name |
| **Dataset** (new `evaluation/FOO/…`) | **add `"FOO"` to `DATASETS`** in `config.py`, or it is silently filtered out | `DATASET_COLORS["FOO"]` (else grey); a `DATASET_NAME_MAP` entry if the folder/file token isn't already the canonical name |

`DATASETS` in `config.py` is an **allow-list**, not just a display-order hint — the loader keeps only datasets on that list. A new dataset that's missing from `DATASETS` won't raise an error; it's just quietly dropped, and the run prints:

```text
[load_results] ignoring datasets outside the report set: [...]
```

so check for that line if a dataset you expect doesn't show up anywhere in the output. Currently configured:

```python
DATASETS = ["HMS", "MME", "LIRICAL", "RAMEDIS", "PUMCH_L", "PUMCH-ADM",
            "0.1.27", "GA4GH_PHENOPACKETS", "MYGENE2_5.7.22", "TEST_MEDICAL_CASES"]
```
