# DX29 Phrank

DX29 Phrank uses the same local Docker container as [DX29 Search](./dx29-search) but queries the `/api/v1/Diagnosis/phrank` endpoint, which implements the [Phrank](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823342/) phenotype ranking algorithm instead of the Dx29 scoring algorithm.

**Complete the setup steps 1–4 from [DX29 Search](./dx29-search) first** — the same Docker container serves both endpoints.


## 1. Running the Benchmark

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


## 2. Difference from DX29 Search

| | DX29 Search | DX29 Phrank |
|---|---|---|
| Endpoint | `/api/v1/Search` | `/api/v1/Diagnosis/phrank` |
| Input format | List of HPO IDs | `{"symptoms": [...], "genes": []}` |
| Algorithm | Dx29 scoring | Phrank |
| Output directory | `dx29_benchmarks/` | `dx29_phrank_benchmarks/` |

Both runners use the same Docker container and share the same setup.

## 3. Implementation
`run_dx29_phrank.py` POSTs `{"symptoms": [...HPO IDs...], "genes": []}` to `/api/v1/Diagnosis/phrank` on the running container (`skip=0`, `count=--topk`, `lang`, `source=all`). The response is a ranked list; the runner reads each entry's `id` and `scoreDx29`, and the first entry whose id exactly matches a confirmed disease gives the case's rank and score. Wall-clock time per request is recorded as `query_time_sec`. Only the output directory and the endpoint differ from DX29 Search — the rest of the pipeline is identical.

> As with DX29 Search, only Orphanet (`ORPHA:`) IDs are returned and matched exactly, so OMIM-only ground truth will not be found.

## 4. Output Format

Results are written to `output/validation_tools/dx29_phrank_benchmarks/<dataset>_summary.tsv` with the same columns as [DX29 Search](./dx29-search#output-format).

## 5. Results

Results across all datasets from the benchmark run (May 2026):

| Dataset |  Found | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Avg. Query Time (s) |
|---------|--------|--------|--------|--------|---------|---------|-------------|
| MME | 34/40 | 0.100 | 0.250 | 0.250 | 0.350 | 0.196 | 0.76 |
| HMS | 65/88 | 0.057 | 0.114 | 0.148 | 0.205 | 0.106 | 0.78 |
| LIRICAL  | 186/370 | 0.108 | 0.151 | 0.184 | 0.230 | 0.147 | 0.77 |
| RAMEDIS  | 292/375 | 0.067 | 0.141 | 0.173 | 0.293 | 0.134 | 0.76 |
| PUMCH_L | 861/988 | 0.118 | 0.200 | 0.228 | 0.293 | 0.180 | 0.85 |
| PUMCH-ADM  | 63/75 | 0.120 | 0.213 | 0.240 | 0.253 | 0.181 | 0.8 |
| GA4GH Phenopackets | 302/384 | 0.148 | 0.216 | 0.253 | 0.315 | 0.205 | 0.78 |
| MyGene2 (5.7.22) | 117/146 | 0.068 | 0.178 | 0.185 | 0.226 | 0.141 | 0.76
| 0.1.27 | 4853/10375 | 0.070 | 0.135 | 0.164| 0.203 | 0.117 | 0.82 |
| test_medical_cases | 192/200 | 0.410 | 0.525 | 0.575 | 0.635 | 0.490 | 0.97 |

> These results use `--topk 1000`. Cases where the ground truth was not found in the top 1000 are counted as not found. DX29 Phrank returns Orphanet IDs only — cases with OMIM-only ground truth will not be matched.


## 6. Reference

> Foundation29. *Dx29.BioNET — Dx29 algorithm for the calculation and suggestion of diseases.* GitHub repository. https://github.com/foundation29org/Dx29.BioNET