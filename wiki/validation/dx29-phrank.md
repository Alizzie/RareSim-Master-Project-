# DX29 Phrank

DX29 Phrank uses the same local Docker container as [DX29 Search](Validation_DX29_Search) but queries the `/api/v1/Diagnosis/phrank` endpoint, which implements the [Phrank](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823342/) phenotype ranking algorithm instead of the Dx29 scoring algorithm.

**Complete the setup steps 1–4 from [DX29 Search](Validation_DX29_Search) first** — the same Docker container serves both endpoints.

---

## Running the Benchmark

```bash
# Run against all datasets (auto-discovered)
python3 run_dx29_phrank.py

# Run against a specific dataset
python3 run_dx29_phrank.py --datasets MME HMS

# Run against a custom dataset directory
python3 run_dx29_phrank.py --data-dir /path/to/your/datasets

# Run against a remote or non-default host
python3 run_dx29_phrank.py --host http://localhost:8080
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--data-dir` | Directory containing dataset JSON files. Default: `datasets/PhenoBrainBenchmarkDatasets`. |
| `--datasets` | Dataset names to run. Default: all JSON files found in `--data-dir`. |
| `--host` | DX29 API base URL. Default: `http://localhost:8080`. |
| `--lang` | Language for API responses. Default: `en`. |
| `--topk` | Number of top predictions to retrieve. Default: `1000`. |

---

## Difference from DX29 Search

| | DX29 Search | DX29 Phrank |
|---|---|---|
| Endpoint | `/api/v1/Search` | `/api/v1/Diagnosis/phrank` |
| Input format | List of HPO IDs | `{"symptoms": [...], "genes": []}` |
| Algorithm | Dx29 scoring | Phrank |
| Output directory | `dx29_benchmarks/` | `dx29_phrank_benchmarks/` |

Both runners use the same Docker container and share the same setup.

---

## Output Format

Results are written to `dx29_phrank_benchmarks/<dataset>_summary.tsv` with the same columns as [DX29 Search](Validation_DX29_Search#output-format).

---

## Results

Results across all datasets from the benchmark run (May 2026):

| Dataset | n | Found | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 | Median rank |
|---------|---|-------|-------|-------|-------|--------|--------|-------------|
| MME | 40 | 34/40 | 0.100 | 0.250 | 0.250 | 0.350 | 0.400 | 57 |
| HMS | 88 | 65/88 | 0.057 | 0.114 | 0.148 | 0.205 | 0.261 | 63 |
| LIRICAL | 370 | 186/370 | 0.108 | 0.151 | 0.184 | 0.230 | 0.268 | 43 |
| RAMEDIS | 375 | 292/375 | 0.067 | 0.141 | 0.173 | 0.293 | 0.384 | 19 |
| PUMCH_L | 988 | 861/988 | 0.118 | 0.200 | 0.228 | 0.293 | 0.350 | 107 |
| PUMCH-ADM | 75 | 63/75 | 0.120 | 0.213 | 0.240 | 0.253 | 0.333 | 86 |

> These results use `--topk 1000`. Cases where the ground truth was not found in the top 1000 are counted as not found. DX29 Phrank returns Orphanet IDs only — cases with OMIM-only ground truth will not be matched.

---

## Reference

> Foundation29. *Dx29.BioNET — Dx29 algorithm for the calculation and suggestion of diseases.* GitHub repository. https://github.com/foundation29org/Dx29.BioNET