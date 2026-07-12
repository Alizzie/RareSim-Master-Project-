# Phenomiser

Phenomiser is a semantic similarity-based tool for rare disease diagnosis prioritisation. Given a set of HPO terms, it ranks candidate diseases by statistical similarity against a precomputed background model.

- **Repository:** [TheJacksonLaboratory/Phenomiser](https://github.com/TheJacksonLaboratory/Phenomiser)
- **Version:** `0.0.2` (commit `eac5fc3`)
- **Tested on:** Ubuntu/Debian, 80 CPUs, 376 GB RAM, May 2026

> **Important:** The `precompute` command was removed in versions after `0.0.2`. You must use the specific commit below — the latest version will not work.

> **Runtime:** Precomputation takes ~20 hours and must be completed once before any queries can be run. Each query takes ~10 minutes.

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Java | 21 | Java 17 will not work |
| Maven | 3.x+ | For building from source |
| RAM | ≥ 32 GB | Required for precompute and queries |
| Disk | ~10 GB | For precomputed cache |

---

## 1. Java Setup

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

---

## 2. Build Phenomiser

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

---

## 3. Input Files

The required input files (`hp.obo` and `phenotype.hpoa`) are already provided in the `phenomiser_data/` folder of this repository. No download or copy step is needed.

```
phenomiser_data/
├── hp.obo
└── phenotype.hpoa        ← already header-fixed (see step 4)
```

> If the files are missing or you need to update them, copy the originals from `ontologies/model/` and re-apply the header fix in step 4. Do not modify the shared files in `ontologies/model/` directly — the `sed` fix in step 4 will break other tools that depend on the original format.

---

## 4. Fix the Annotation File Header

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

---

## 5. Precomputation

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

---

## 6. Running the Benchmark

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

---

## Output Format

Results are written to `phenomizer_benchmarks/<dataset>_summary.tsv`:

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

Raw per-case output files (`.txt`) are cached in `phenomizer_benchmarks/cache/<dataset>/`.

### Phenomiser output columns

| Column | Description |
|--------|-------------|
| `diseaseId` | OMIM identifier |
| `diseaseName` | Disease name |
| `p` | Raw p-value (lower = better match) |
| `adjust_p` | Adjusted p-value (multiple testing corrected) |
| `similarityScore` | Semantic similarity score (higher = more similar) |

Results are sorted by adjusted p-value ascending — the top rows are the best candidate diagnoses.

---

## Performance

| Step | Runtime (40 threads) |
|------|----------------------|
| Precomputation | ~20 hours |
| Per-case query | ~10 minutes |

Plan accordingly for large datasets — 500 cases will take approximately 3–4 days of query time running sequentially. Use `--skip-existing` to safely resume interrupted runs.

## References

> Köhler S. et al. *Clinical diagnostics in human genetics with semantic similarity searches in ontologies.* Am. J. Hum. Genet. 85, 457–464 (2009). https://doi.org/10.1016/j.ajhg.2009.09.003