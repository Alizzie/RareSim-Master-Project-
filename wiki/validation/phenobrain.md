# PhenoBrain

PhenoBrain is an ensemble AI model for rare disease diagnosis prioritisation. It accepts HPO terms as input and returns ranked candidate diseases via a web API — no local installation or precomputation required.

- **Repository:** [xiaohaomao/timgroup_disease_diagnosis](https://github.com/xiaohaomao/timgroup_disease_diagnosis)
- **API documentation:** [PhenoBrain Web API README](https://github.com/xiaohaomao/timgroup_disease_diagnosis/blob/main/PhenoBrain_Web_API/README.md)
- **Web interface:** [phenobrain.cs.tsinghua.edu.cn](http://www.phenobrain.cs.tsinghua.edu.cn/pc)
- **Tested on:** macOS, May 2026

> **Note:** PhenoBrain runs as a hosted service. It covers the hosted PhenoBrain API, so there is no local setup required. Queries are sent to the Tsinghua University API server. Results depend on the availability of that server.
For the self-hosted deployment (running their GitHub pipeline locally and standardizing its raw CSV), see the [PhenoBrain (Local)](./phenobrain-local.md) page.

## 1. Requirements

- Python 3.9+
- `requests` library (`pip install requests`)
- Internet access to reach `www.phenobrain.cs.tsinghua.edu.cn`

## 2. How It Works

PhenoBrain uses an asynchronous task-based API. Each query follows a two-step process:

1. Submit a prediction request with a list of HPO terms → receive a `TASK_ID`
2. Poll the result endpoint with the `TASK_ID` until the state is `SUCCESS`

The runner handles this automatically, polling every 3 seconds with a 300-second timeout per case.

The returned disease codes use PhenoBrain's internal `RD:` namespace. A second API call maps these to standard OMIM and Orphanet identifiers for comparison against ground truth.


## 3. Running the Benchmark

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

## 4. Implementation
`run_phenobrain.py` processes each case against the live API:

1. **Submit:** POSTs the case's HPO terms to `/predict` (Ensemble model) and receives a `TASK_ID` (with up to 3 retries on transient failures).
2. **Poll:** Polls `/query-predict-result` every 3 s (300 s timeout) until the task state is `SUCCESS`.
3. **Remap:** The results carry internal `RD`: codes; a batched `/disease-list-detail` call maps them to OMIM/ORPHA before matching against ground truth.
4. **Rank:** The best (lowest) rank whose mapped codes contain a confirmed disease is taken as the case result, then written into the summary.

`query_time_sec` covers the full submit + poll + remap round-trip. There is no local cache; every run re-queries the API.

## 5. Output Format

Results are written to `output/validation_tools/phenobrain_benchmarks/<dataset>_summary.tsv`:

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

## 6. API Notes

The PhenoBrain API uses internal `RD:` and `CCRD:` disease codes rather than OMIM or Orphanet IDs directly. The runner resolves these to standard identifiers via a `disease-list-detail` API call before comparing against ground truth.

The available prediction models are: `Ensemble`, `ICTO (A)`, `ICTO (U)`, `PPO`, `CNB`, `MLP (M)`, `MinIC`, `Res`, `BOQA`, `GDDP`, `RBP`, `Lin`, `JC`, `SimUI`, `TO`, `Cosine`, `RDD`. The benchmark uses `Ensemble` as it is the top-performing model.

## 7. Performance

| Metric | Value |
|--------|-------|
| Per-case query time | ~5–6 seconds |
| Includes API round-trip and result polling | yes |

Query time is dominated by network latency and server processing. There is no `--skip-existing` option since results are not cached locally — re-running will re-query the API.

## 8. Results

Results across all datasets from the benchmark run (May 2026):

| Dataset |  Found | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Avg. Query Time (s) |
|---------|--------|--------|--------|--------|---------|---------|-------------|
| MME |  39/40 | 0.500 | 0.725 | 0.800 | 0.850 | 0.634 | 5.6 |
| HMS |  80/88 | 0.193 | 0.330 | 0.409 | 0.523 | 0.296 | 5.6 |
| LIRICAL |  344/370 | 0.341 | 0.481 | 0.532 | 0.603 | 0.431 | 5.6 |
| RAMEDIS |  341/375 | 0.272 | 0.493 | 0.563 | 0.659 | 0.409 | 5.6 |
| PUMCH_L |  921/988 | 0.306 | 0.465 | 0.545 | 0.633 | 0.418 | 5.6 |
| PUMCH-ADM | 74/75 | 0.400 | 0.573 | 0.627 | 0.680 | 0.510 | 5.6 |
| GA4GH Phenopackets | 362/384 | 0.346 | 0.487 | 0.544 | 0.641 | 0.443 | 6.25 |
| MyGene2 (5.7.22) | 131/146 | 0.363 | 0.555 | 0.589 | 0.637 | 0.463 | 6.25 |
| 0.1.27 | 5991/10374 | 0.176 | 0.270 | 0.306 | 0.365 | 0.241 | 5.65 |
| test_medical_cases | 195/200 | 0.775 | 0.870 | 0.890 | 0.915 | 0.830 | 6.26 |

> These results use the Ensemble model with `--topk 200`. Cases where the ground truth was not in the top 200 are counted as not found.


## 9. Reference

> Mao X. et al. *A phenotype-based AI pipeline outperforms human experts in differentially diagnosing rare diseases using EHRs.* npj Digital Medicine 8, 68 (2025). https://doi.org/10.1038/s41746-025-01452-1