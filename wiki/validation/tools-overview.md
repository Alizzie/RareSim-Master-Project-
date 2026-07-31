# Validation Tools Overview

This folder contains benchmark runners and results for comparing existing rare disease diagnostic tools against curated patient datasets. The goal is to establish a performance baseline for comparisong with our own benchmarking tool.

Each tool has its own runner script and produces standardised output, making it straightforward to add new tools or datasets in the future.


## 1. Tools

| Tool | Script |
|------|--------|
| [LIRICAL](./lirical) | `run_lirical.py` | 
| [PhenoBrain](./phenobrain) | `run_phenobrain.py` | 
| [PhenoBrain (Local)](./phenobrain-local.md) | `run_phenobrain_local.py` |
| [Phenomiser](./phenomiser) | `run_phenomiser.py` | 
| [DX29 Search](./dx29-search) | `run_dx29_search.py` | 
| [DX29 Phrank](./dx29-phrank) | `run_dx29_phrank.py` | 


## 2. Datasets

All tools are run against the PhenoBrainBenchmarkDatasets. Dataset JSON files are located in datasets/PhenoBrainBenchmarkDatasets/. See Dataset Format for the expected file structure and how to add new datasets.

## 3. Workflow

```
datasets/
  └── PhenoBrainBenchmarkDatasets/*.json
          │
          ▼
  run_<tool>.py          (one per tool)
          │
          ▼
  <tool>_benchmarks/
    ├── cache/           (raw per-case output)
    └── <dataset>_summary.tsv
          │
          ▼
  compare_methods.py
          │
          ▼
  results/
    └── comparison_summary.tsv
```

1. Run each tool against the desired dataset(s) — outputs a TSV summary per dataset.
2. Run `compare_methods.py` to aggregate all summaries into a single comparison report.

> PhenoBrain (Local) is the exception to step 1: you run the upstream pipeline yourself to get a raw CSV, then `run_phenobrain_local.py` . standardizes it, by default one `output/validation_tools/phenobrain (<model>)_benchmarks/` folder per model column, so every model shows up as its own method. 


## 4. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single tool against all datasets (auto-discovered)
python run_lirical.py \
  --lirical-jar path/to/lirical-cli.jar \
  --lirical-data ~/lirical-data

# Run against a specific dataset
python run_lirical.py \
  --lirical-jar path/to/lirical-cli.jar \
  --lirical-data ~/lirical-data \
  --datasets HMS MME

# Run against a custom dataset directory
python run_lirical.py \
  --lirical-jar path/to/lirical-cli.jar \
  --lirical-data ~/lirical-data \
  --data-dir /path/to/your/datasets

# Compare all tool results (no flags needed — discovery is automatic)
python compare_methods.py
```

See each tool's documentation page for full argument references and setup instructions.


## 5. Output Format

Each runner writes a TSV summary file to its benchmark directory (e.g. `output/validation_tools/lirical_benchmarks/<dataset>_summary.tsv`) with the following columns:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Expected disease ID(s) |
| `rank` | Rank of correct diagnosis (`None` if not found) |
| `matched_id` | Disease ID that matched |
| `score` | Tool-specific confidence score |
| `status` | Whether the tool ran successfully |
| `query_time_sec` | Time taken for the query |


## 6. Repository Structure

```
data/
├── datasets/ # Input patient case datasets

validation_tools/
├── phenobrain_raw                     # xlsx raw output files from phenobrain local
├── _utils.py                          # Shared dataset loading and statistics
├── compare_methods.py                 # Cross-tool comparison script
├── run_lirical.py
├── run_phenobrain.py
├── run_phenobrain_local.py            # Standardizes local PhenoBrain raw CSVs
├── run_phenomiser.py
├── run_dx29_search.py
└── run_dx29_phrank.py

output/validation_tools/               # All generated outputs live here
├── lirical_benchmarks/                # LIRICAL cache and summaries
├── phenobrain_benchmarks/             # Hosted PhenoBrain summaries
├── phenobrain (<model>)_benchmarks/   # Local PhenoBrain, one folder per model 
├── phenomizer_benchmarks/             # Phenomiser cache and summaries
├── dx29_benchmarks/                   # DX29 Search cache and summaries
├── dx29_phrank_benchmarks/            # DX29 Phrank cache and summaries
└── results/                           # Aggregated comparison results
```

## 7. Extending with a New Tool

1. Add a new runner script following the pattern of any existing `run_<tool>.py`.
2. Import `resolve_datasets` and `load_all_datasets` from `utils.py`; dataset discovery is handled automatically.
3. Write output to a new `<tool>_benchmarks/` directory using `save_summary_tsv` from `utils.py`.
4. `compare_methods.py` auto-discovers any `*_benchmarks/` folder; no registration step is needed
5. Add a documentation page and link it in the sidebar.
