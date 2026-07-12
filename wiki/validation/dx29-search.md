# DX29 Search

DX29 BioNET is a .NET-based rare disease prioritisation service developed by Foundation29. It exposes a local REST API that ranks candidate diseases from HPO terms using its own scoring algorithm. Two endpoints are benchmarked: the Dx29 Search algorithm (`/api/v1/Search`) and the Phrank algorithm (`/api/v1/Diagnosis/phrank`) — each has its own runner and results page.

- **Repository:** [foundation29org/Dx29.BioNET](https://github.com/foundation29org/Dx29.BioNET)
- **Documentation:** [Dx29 v2 — Running locally](https://dx29-v2.readthedocs.io/en/latest/pages/4_BuildAndDeploy/4_Locally.html)
- **Tested on:** macOS / Linux, May 2026

> **Note:** DX29 runs as a local Docker container. The API server must be running before executing the benchmark runner.

---

## Requirements

| Dependency | Notes |
|------------|-------|
| Docker Desktop | Required to build and run the container |
| Python 3.9+ | For the benchmark runner |
| ~500 MB disk | For the OrphaNet XML data files |

---

## 1. Download Data Files

The repository includes truncated OrphaNet data files due to Git file size limits. Replace them with the full versions before building.

From the `src/` directory of the cloned repository:

```bash
# HPO phenotype annotations (~46 MB)
curl -L -o ./Dx29.BioNET.WebAPI/_data/orpha-phen.xml \
  "http://www.orphadata.org/data/xml/en_product4.xml"

# Gene-disease associations
curl -L -o ./Dx29.BioNET.WebAPI/_data/orpha-gene.xml \
  "http://www.orphadata.org/data/xml/en_product6.xml"
```

Verify the download:

```bash
ls -lh ./Dx29.BioNET.WebAPI/_data/
# orpha-phen.xml should be ~46 MB — if it is only a few MB, the download failed
```

---

## 2. Apply Code Fix

The original code throws runtime exceptions when the ORDO ontology contains duplicate disease IDs or blank cross-reference keys. Apply this patch to `src/Dx29.BioNET/OrphaNET/OrphaNET.Ordo.cs` before building:

```csharp
// Before
var xrefs = GetXRefs(node).ToDictionary(r => r.Item1, r => r.Item2);
var disease = new Disease(id, name) { ... };
Diseases.Add(id, disease);

// After
var xrefs = GetXRefs(node)
    .Where(r => !string.IsNullOrWhiteSpace(r.Item1))
    .GroupBy(r => r.Item1.Trim())
    .ToDictionary(g => g.Key, g => g.First().Item2);
var disease = new Disease(id, name) { ... };
if (!Diseases.ContainsKey(id))
{
    Diseases.Add(id, disease);
}
```

This prevents two errors: `ArgumentException: An item with the same key has already been added` caused by duplicate disease IDs, and the same exception caused by blank XRef keys.

---

## 3. Build the Docker Image

From the `src/` directory:

```bash
docker build -t dx29-bionet -f Dx29.BioNET.WebAPI/Dockerfile .
```

The first build takes a few minutes as it pulls the .NET 5 base images.

---

## 4. Run the Container

```bash
docker run -d -p 8080:80 --name dx29-bionet dx29-bionet:latest
```

The app loads the OrphaNet XML files on startup — allow 20–30 seconds before running queries. Monitor with:

```bash
docker logs dx29-bionet
```

The API is then available at `http://localhost:8080`, with Swagger UI at `http://localhost:8080/swagger`.

---

## 5. Running the Benchmark

```bash
# Run against all datasets (auto-discovered)
python3 run_dx29_search.py

# Run against a specific dataset
python3 run_dx29_search.py --datasets MME HMS

# Run against a custom dataset directory
python3 run_dx29_search.py --data-dir /path/to/your/datasets

# Run against a remote or non-default host
python3 run_dx29_search.py --host http://localhost:8080
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

## Output Format

Results are written to `dx29_benchmarks/<dataset>_summary.tsv`:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Matched disease ID from the DX29 response |
| `rank` | Rank of correct diagnosis (`None` if not found in top `--topk`) |
| `matched_id` | Disease ID that matched the ground truth |
| `score` | DX29 similarity score |
| `status` | Whether the API call succeeded |
| `query_time_sec` | Time taken for the query |

> **Note:** DX29 uses Orphanet disease IDs (`ORPHA:`) directly. Ground truth IDs are matched exactly — cases with OMIM-only ground truth may not be found even if the disease is ranked.

---

## Results

Results across all datasets from the benchmark run (May 2026):

| Dataset | n | Found | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 | Median rank |
|---------|---|-------|-------|-------|-------|--------|--------|-------------|
| MME | 40 | 36/40 | 0.425 | 0.600 | 0.650 | 0.700 | 0.725 | 3 |
| HMS | 88 | 80/88 | 0.205 | 0.341 | 0.364 | 0.500 | 0.568 | 12 |
| LIRICAL | 370 | 193/370 | 0.173 | 0.216 | 0.257 | 0.305 | 0.362 | 8 |
| RAMEDIS | 375 | 302/375 | 0.088 | 0.173 | 0.253 | 0.373 | 0.477 | 15 |
| PUMCH_L | 988 | 891/988 | 0.211 | 0.338 | 0.401 | 0.491 | 0.592 | 5 |
| PUMCH-ADM | 75 | 70/75 | 0.173 | 0.267 | 0.360 | 0.413 | 0.533 | 17 |

> These results use `--topk 1000`. Cases where the ground truth was not found in the top 1000 are counted as not found. DX29 returns Orphanet IDs only — cases with OMIM-only ground truth will not be matched.


---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `XmlException: Unexpected end of file` | Truncated `orpha-phen.xml` | Re-download (step 1) |
| `ArgumentException: duplicate key` | Code fix not applied | Apply the patch (step 2) |
| Empty logs, container running | App still loading XML | Wait 20–30 s and retry |
| `Connection reset by peer` | App crashed on startup | Run `docker logs dx29-bionet` |

---

## Reference

> Foundation29. *Dx29.BioNET — Dx29 algorithm for the calculation and suggestion of diseases.* GitHub repository. https://github.com/foundation29org/Dx29.BioNET