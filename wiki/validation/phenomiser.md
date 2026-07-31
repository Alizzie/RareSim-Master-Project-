# Phenomiser

Phenomiser is a semantic similarity-based tool for rare disease diagnosis prioritisation. Given a set of HPO terms, it ranks candidate diseases by statistical similarity against a precomputed background model.

- **Repository:** [TheJacksonLaboratory/Phenomiser](https://github.com/TheJacksonLaboratory/Phenomiser)
- **Version:** `0.0.2` (commit `eac5fc3`)
- **Tested on:** Ubuntu/Debian, 80 CPUs, 376 GB RAM, May 2026

> **Important:** The `precompute` command was removed in versions after `0.0.2`. You must use the specific commit below — the latest version will not work.

> **Runtime:** Precomputation takes ~20 hours and must be completed once before any queries can be run. Each query takes ~10 minutes.

## 1. Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Java | 21 | Java 17 will not work |
| Maven | 3.x+ | For building from source |
| RAM | ≥ 32 GB | Required for precompute and queries |
| Disk | ~10 GB | For precomputed cache |


## 2. Java Setup

Phenomiser requires Java 21. Verify your version:

```bash
java -version
# Should show: openjdk version "21..."
```

If Maven is using a different Java version, set `JAVA_HOME` explicitly:

```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export PATH=$JAVA_HOME/bin:$PATH

# Verify
mvn -version
# Should show: Java version: 21...
```

## 3. Build Phenomiser

Clone the repository and check out the correct commit:

```bash
git clone https://github.com/TheJacksonLaboratory/Phenomiser.git
cd Phenomiser
git checkout eac5fc3
```

Build the JAR (tests are skipped as they require additional data):

```bash
mvn package -DskipTests
```

The JAR will be at:

```
phenomiser-cli/target/phenomiser-cli-0.0.2.jar
```

## 4. Input Files

The required input files (`hp.obo` and `phenotype.hpoa`) are already provided in the `phenomiser_data/` folder of this repository. No download or copy step is needed.

```
phenomiser_data/
├── hp.obo
└── phenotype.hpoa        ← already header-fixed (see step 4)
```

> If the files are missing or you need to update them, copy the originals from `ontologies/model/` and re-apply the header fix in step 4. Do not modify the shared files in `ontologies/model/` directly — the `sed` fix in step 4 will break other tools that depend on the original format.


## 5. Fix the Annotation File Header

> **Note:** This step is already done for the file in `phenomiser_data/`. Only follow these instructions if you are setting up a fresh copy.

The current `phenotype.hpoa` format uses lowercase field names, but Phenomiser `0.0.2` expects camelCase headers. Apply the following fix to the copy in `phenomiser_data/`:

```bash
sed -i 's/database_id/DatabaseID/g; \
        s/disease_name/DiseaseName/g; \
        s/qualifier/Qualifier/g; \
        s/hpo_id/HPO_ID/g; \
        s/reference/Reference/g; \
        s/evidence/Evidence/g; \
        s/onset/Onset/g; \
        s/frequency/Frequency/g; \
        s/sex/Sex/g; \
        s/modifier/Modifier/g; \
        s/aspect/Aspect/g; \
        s/biocuration/Biocuration/g' \
        phenomiser_data/phenotype.hpoa
```

Verify the header is correct:

```bash
head -6 phenomiser_data/phenotype.hpoa
# First data line should start with: DatabaseID  DiseaseName  ...
```


## 6. Precomputation

Phenomiser must precompute a statistical background model before queries can be run. This step is slow but only needs to be done once per machine.

### Test first (debug mode)

Debug mode only processes 50 diseases and completes in a few minutes. Always run this first to confirm the setup is correct:

```bash
java -Xmx32g \
  -jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar precompute \
  -hpo ~/phenomiser_data/hp.obo \
  -da ~/phenomiser_data/phenotype.hpoa \
  -debug \
  -numThreads 4
```

If it outputs a ranked disease table, the setup is correct.

### Full precomputation

Run inside a `screen` session so it continues if you disconnect:

```bash
screen -S phenomiser

java -Xmx32g \
  -jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar precompute \
  -hpo ~/phenomiser_data/hp.obo \
  -da ~/phenomiser_data/phenotype.hpoa \
  --sampling 1 10 \
  -numThreads 40
```

| Action | Command |
|--------|---------|
| Detach from screen | `Ctrl+A, D` |
| Reattach later | `screen -r phenomiser` |
| List running screens | `screen -ls` |

Expected runtime: ~20 hours with 40 threads. The cache is saved to `~/Phenomiser_data/` automatically.


## 7. Running the Benchmark

Once precomputation is complete, run the benchmark against your datasets:

```bash
# Run against all datasets (auto-discovered)
python3 run_phenomiser.py \
  --phenomizer-jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar \
  --hp-obo ~/phenomiser_data/hp.obo \
  --hpoa ~/phenomiser_data/phenotype.hpoa

# Run against a specific dataset
python3 run_phenomiser.py \
  --phenomizer-jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar \
  --hp-obo ~/phenomiser_data/hp.obo \
  --hpoa ~/phenomiser_data/phenotype.hpoa \
  --datasets MME HMS

# Run against a custom dataset directory
python3 run_phenomiser.py \
  --phenomizer-jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar \
  --hp-obo ~/phenomiser_data/hp.obo \
  --hpoa ~/phenomiser_data/phenotype.hpoa \
  --data-dir /path/to/your/datasets

# Resume an interrupted run
python3 run_phenomiser.py \
  --phenomizer-jar ~/Phenomiser/phenomiser-cli/target/phenomiser-cli-0.0.2.jar \
  --hp-obo ~/phenomiser_data/hp.obo \
  --hpoa ~/phenomiser_data/phenotype.hpoa \
  --skip-existing
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--phenomizer-jar` | Path to Phenomiser JAR. Required. |
| `--hp-obo` | Path to `hp.obo`. Required. |
| `--hpoa` | Path to `phenotype.hpoa` (the modified copy in `phenomiser_data/`). Required. |
| `--data-dir` | Directory containing dataset JSON files. Default: `datasets/PhenoBrainBenchmarkDatasets`. |
| `--datasets` | Dataset names to run. Default: all JSON files found in `--data-dir`. |
| `--java` | Path to Java 21 executable. Default: `java`. |
| `--xmx` | Java heap size. Default: `32g`. |
| `--skip-existing` | Skip cases with existing output (resume mode). Default: off. |

## 8. Implementations
`run_phenomiser.py` runs in two phases per dataset:

1. **Run:** For each case it invokes the Phenomiser CLI `query` subcommand once (comma-separated HPO terms via `-query`), writing one `<case_id>.txt` into `cache/<dataset>/`. Wall-clock time per subprocess call is recorded as `query_time_sec`; `--skip-existing` skips cases whose output already exists.
2. **Collect:** Each cached file is parsed in file order (Phenomiser sorts by adjusted p-value ascending, then similarity score descending) and ranks are assigned accordingly. The best (lowest) rank among the confirmed disease IDs is taken as the case result, then rolled into the summary and statistics.


## 9. Output Format

Results are written to `output/validation_tools/phenomizer_benchmarks/<dataset>_summary.tsv`:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Expected disease ID(s) |
| `rank` | Rank of correct diagnosis (`None` if not found) |
| `matched_id` | Disease ID that matched |
| `score` | Semantic similarity score |
| `status` | Whether the tool ran successfully |
| `query_time_sec` | Time taken for the query |

Raw per-case output files (`.txt`) are cached in `output/validation_tools/phenomizer_benchmarks/cache/<dataset>/`.

### Phenomiser output columns

| Column | Description |
|--------|-------------|
| `diseaseId` | OMIM identifier |
| `diseaseName` | Disease name |
| `p` | Raw p-value (lower = better match) |
| `adjust_p` | Adjusted p-value (multiple testing corrected) |
| `similarityScore` | Semantic similarity score (higher = more similar) |

Results are sorted by adjusted p-value ascending — the top rows are the best candidate diagnoses.

## 10. Performance

| Step | Runtime (40 threads) |
|------|----------------------|
| Precomputation | ~20 hours |
| Per-case query | ~10 minutes |

Plan accordingly for large datasets — 500 cases will take approximately 3–4 days of query time running sequentially. Use `--skip-existing` to safely resume interrupted runs.


## 11. Results

Results across all datasets from the benchmark run (Juni 2026):

| Dataset |  Found | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Avg. Query Time (s) |
|---------|--------|--------|--------|--------|---------|---------|-------------|
| MME  | 40/40 | 0.425 | 0.550 | 0.575 | 0.675 | 0.500 | 1297.75 |
| HMS |  62/88 | 0.080 | 0.159 | 0.182 | 0.205 | 0.130 | 1107.25 |
| LIRICAL  | 367/370 | 0.276 | 0.411 | 0.459 | 0.522 | 0.536 | 1488.01 |
| RAMEDIS | 375/375 | 0.085 | 0.203 | 0.203 | 0.248 | 0.147 | 1325.87 |
| PUMCH_L  |720/988 | 0.167 | 0.252 | 0.286 | 0.359 | 0.229 | 990.05 | 
| PUMCH-ADM | 68/75 | 0.133 | 0.240 | 0.333 | 0.400 | 0.224 | 2004.54 | 
| GA4GH Phenopackets | 384/384 | 0.266 | 0.409 | 0.456 | 0.526 | 0.358 | 1793.54 |
| MyGene2 (5.7.22) | 146/146 | 0.226 | 0.315 | 0.336 | 0.500 | 0.298 | 2191.66 |
| 0.1.27 | / | / | / | / | / | / | / |
| test_medical_cases | 0/200 | 0 | 0 | 0 | 0 | 0 | 2049.25 |

**Note:** Results for the dataset `0.1.27` are unavailable because the dataset is substantially larger than the others. Based on current performance, processing the full dataset is estimated to require more than two months of sequential computation and was therefore not completed.

## 12. References

> Köhler S. et al. *Clinical diagnostics in human genetics with semantic similarity searches in ontologies.* Am. J. Hum. Genet. 85, 457–464 (2009). https://doi.org/10.1016/j.ajhg.2009.09.003