# PhenoBrain

PhenoBrain is an ensemble AI model for rare disease diagnosis prioritisation. It accepts HPO terms as input and returns ranked candidate diseases via a web API — no local installation or precomputation required.

- **Repository:** [xiaohaomao/timgroup_disease_diagnosis](https://github.com/xiaohaomao/timgroup_disease_diagnosis)
- **API documentation:** [PhenoBrain Web API README](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/PhenoBrain_Web_API/README.md)
- **Web interface:** [phenobrain.cs.tsinghua.edu.cn](http://www.phenobrain.cs.tsinghua.edu.cn/pc)
- **Tested on:** macOS, May 2026

> **Note:** PhenoBrain runs as a hosted service — there is no local setup required. Queries are sent to the Tsinghua University API server. Results depend on the availability of that server.

---

## Requirements

- Python 3.9+
- `requests` library (`pip install requests`)
- Internet access to reach `www.phenobrain.cs.tsinghua.edu.cn`

---

## How It Works

PhenoBrain uses an asynchronous task-based API. Each query follows a two-step process:

1. Submit a prediction request with a list of HPO terms → receive a `TASK_ID`
2. Poll the result endpoint with the `TASK_ID` until the state is `SUCCESS`

The runner handles this automatically, polling every 3 seconds with a 300-second timeout per case.

The returned disease codes use PhenoBrain's internal `RD:` namespace. A second API call maps these to standard OMIM and Orphanet identifiers for comparison against ground truth.

---

## Running the Benchmark

```bash
# Run against all datasets (auto-discovered)
python3 run_phenobrain.py

# Run against a specific dataset
python3 run_phenobrain.py --datasets MME HMS

# Run against a custom dataset directory
python3 run_phenobrain.py --data-dir /path/to/your/datasets
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--data-dir` | Directory containing dataset JSON files. Default: `datasets/PhenoBrainBenchmarkDatasets`. |
| `--datasets` | Dataset names to run. Default: all JSON files found in `--data-dir`. |
| `--topk` | Number of top predictions to retrieve. Maximum and default: `200`. |

> **Note:** The maximum value for `--topk` is 200. Values above 200 are silently capped to 200 by the API. Cases where the correct diagnosis falls outside the top 200 will have `rank = None`.

---

## Output Format

Results are written to `phenobrain_benchmarks/<dataset>_summary.tsv`:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Matched disease ID(s) from PhenoBrain's disease mapping (OMIM/ORPHA/CCRD) |
| `rank` | Rank of correct diagnosis (`None` if not found in top 200) |
| `matched_id` | Disease ID that matched the ground truth |
| `score` | Ensemble similarity score (0–1, higher = better match) |
| `status` | Whether the API call succeeded |
| `query_time_sec` | Total time including API polling |

Raw per-case output is not cached — each run queries the live API. Use the summary TSV as the persistent record of results.

---

## Results

Results across all datasets from the benchmark run (May 2026):

| Dataset | n | Found | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 | Median rank |
|---------|---|-------|-------|-------|-------|--------|--------|-------------|
| MME | 40 | 39/40 | 0.500 | 0.725 | 0.800 | 0.850 | 0.875 | 1 |
| HMS | 88 | 80/88 | 0.193 | 0.330 | 0.409 | 0.523 | 0.659 | 7.5 |
| LIRICAL | 370 | 344/370 | 0.341 | 0.481 | 0.532 | 0.603 | 0.662 | 3.0 |
| RAMEDIS | 375 | 341/375 | 0.272 | 0.493 | 0.563 | 0.659 | 0.741 | 3 |
| PUMCH_L | 988 | 921/988 | 0.306 | 0.465 | 0.545 | 0.633 | 0.705 | 4 |
| PUMCH-ADM | 75 | 74/75 | 0.400 | 0.573 | 0.627 | 0.680 | 0.773 | 2.0 |

> These results use the Ensemble model with `--topk 200`. Cases where the ground truth was not in the top 200 are counted as not found.

---

## API Notes

The PhenoBrain API uses internal `RD:` and `CCRD:` disease codes rather than OMIM or Orphanet IDs directly. The runner resolves these to standard identifiers via a `disease-list-detail` API call before comparing against ground truth.

The available prediction models are: `Ensemble`, `ICTO (A)`, `ICTO (U)`, `PPO`, `CNB`, `MLP (M)`, `MinIC`, `Res`, `BOQA`, `GDDP`, `RBP`, `Lin`, `JC`, `SimUI`, `TO`, `Cosine`, `RDD`. The benchmark uses `Ensemble` as it is the top-performing model.

---

## Performance

| Metric | Value |
|--------|-------|
| Per-case query time | ~5–6 seconds |
| Includes API round-trip and result polling | yes |

Query time is dominated by network latency and server processing. There is no `--skip-existing` option since results are not cached locally — re-running will re-query the API.

---

## Reference

> Mao X. et al. *A phenotype-based AI pipeline outperforms human experts in differentially diagnosing rare diseases using EHRs.* npj Digital Medicine 8, 68 (2025). https://doi.org/10.1038/s41746-025-01452-1