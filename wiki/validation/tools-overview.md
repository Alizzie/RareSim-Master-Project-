# Validation Tools Overview

This folder contains benchmark runners and results for comparing existing rare disease diagnostic tools against curated patient datasets. The goal is to establish a performance baseline for comparisong with our own benchmarking tool.

Each tool has its own runner script and produces standardised output, making it straightforward to add new tools or datasets in the future.


## Tools

| Tool | Script |
|------|--------|
| [LIRICAL](./lirical) | `run_lirical.py` | 
| [PhenoBrain](./phenobrain) | `run_phenobrain.py` | 
| [Phenomiser](./phenomiser) | `run_phenomiser.py` | 
| [DX29 Search](./dx29-search) | `run_dx29_search.py` | 
| [DX29 Phrank](./dx29-phrank) | `run_dx29_phrank.py` | 

---

## Datasets

All tools are run against the PhenoBrainBenchmarkDatasets. Dataset JSON files are located in datasets/PhenoBrainBenchmarkDatasets/. See Dataset Format for the expected file structure and how to add new datasets.

---

## Workflow

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

---

## Quick Start

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

# Compare all tool results
python compare_methods.py --results results/
```

See each tool's documentation page for full argument references and setup instructions.

---

## Output Format

Each runner writes a TSV summary file to its benchmark directory (e.g. `lirical_benchmarks/<dataset>_summary.tsv`) with the following columns:

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

---

## Repository Structure

```
validation_tools/
├── datasets/
│   └── PhenoBrainBenchmarkDatasets/   # Input patient case datasets (JSON)
├── lirical_benchmarks/                # LIRICAL cache and summaries
├── phenobrain_benchmarks/             # PhenoBrain cache and summaries
├── phenomizer_benchmarks/             # Phenomiser cache and summaries
├── dx29_benchmarks/                   # DX29 Search cache and summaries
├── dx29_phrank_benchmarks/            # DX29 Phrank cache and summaries
├── results/                           # Aggregated comparison results
├── compare_methods.py                 # Cross-tool comparison script
├── utils.py                           # Shared dataset loading and statistics
├── run_lirical.py
├── run_phenobrain.py
├── run_phenomiser.py
├── run_dx29_search.py
└── run_dx29_phrank.py
```

---

## Extending with a New Tool

1. Add a new runner script following the pattern of any existing `run_<tool>.py`.
2. Import `resolve_datasets` and `load_all_datasets` from `utils.py` — dataset discovery is handled automatically.
3. Write output to a new `<tool>_benchmarks/` directory using `save_summary_tsv` from `utils.py`.
4. Register the tool in `compare_methods.py`.
5. Add a documentation page and link it in the sidebar.
