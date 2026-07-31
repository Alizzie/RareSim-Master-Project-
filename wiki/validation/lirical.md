# LIRICAL

LIRICAL (LIkelihood Ratio Interpretation of Clinical AbnormaLities) is a phenotype-driven disease prioritisation tool that uses a likelihood ratio framework to rank candidate diagnoses given a set of HPO terms.

- **Repository:** [TheJacksonLaboratory/LIRICAL](https://github.com/TheJacksonLaboratory/LIRICAL)
- **Documentation:** [LIRICAL Setup Guide](https://thejacksonlaboratory.github.io/LIRICAL/stable/setup.html)
- **Version:** `2.4.0`
- **Tested on:** macOS, May 2026

> **Important:** The YAML input format changed between versions. In v2.4.0, the `yaml` subcommand does not work correctly. The runner uses the `prioritize` subcommand with `-p` flags instead.


## 1. Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Java | 17+ | Verify with `java -version` |
| RAM | 8 GB+ | Recommended |
| Disk | ~500 MB | For LIRICAL data directory |


## 2. Install LIRICAL

Clone the repository and build using the Maven wrapper. Prior installation of Maven is not required.

```bash
git clone https://github.com/TheJacksonLaboratory/LIRICAL.git
cd LIRICAL
./mvnw -Prelease install
```

The distribution ZIP will be at:

```
lirical-cli/target/lirical-cli-2.4.0-distribution.zip
```

Unzip it:

```bash
cd lirical-cli/target/
unzip lirical-cli-2.4.0-distribution.zip
```

The runnable JAR will be at:

```
lirical-cli/target/lirical-cli-2.4.0/lirical-cli-2.4.0.jar
```

Set up an alias for convenience:

```bash
alias lirical="java -jar /path/to/lirical-cli-2.4.0.jar"

# Verify
lirical --version
```

To make the alias permanent, add it to your `~/.zshrc` or `~/.bashrc`.

> **Note:** If your path contains spaces (e.g. `Master Project/`), quote the full path when referencing the JAR.

> Alternatively, a prebuilt executable is available from the [Releases page](https://github.com/TheJacksonLaboratory/LIRICAL/releases). See the [official setup guide](https://thejacksonlaboratory.github.io/LIRICAL/stable/setup.html) for details.


## 3. Data Files

LIRICAL requires a data directory containing the HPO ontology and gene/transcript annotation files. Run the built-in download command once:

```bash
cd lirical-cli-2.4.0
lirical download
```

> This downloads ~500 MB and only needs to be done once. For phenotype-only mode (no VCF), Exomiser variant databases are not needed. See the [official setup guide](https://thejacksonlaboratory.github.io/LIRICAL/stable/setup.html) for details.
> If you want the data in a different location, you need to pass that path explicitly when running the benchmark via --lirical-data.


## 4. Verify Installation

Test with a single case using the `prioritize` subcommand:

```bash
java -jar /path/to/lirical-cli-2.4.0.jar prioritize \
  -p HP:0002321,HP:0000365,HP:0000360,HP:0009589,HP:0002858 \
  -n HP:0009736 \
  -d lirical_data/
```

If it outputs a ranked disease table, the setup is correct.

## 5. Running the Benchmark

```bash
# Run against all datasets (auto-discovered)
python3 run_lirical.py \
  --lirical-jar /path/to/lirical-cli-2.4.0.jar \
  --lirical-data lirical_data/

# Run against a specific dataset
python3 run_lirical.py \
  --lirical-jar /path/to/lirical-cli-2.4.0.jar \
  --lirical-data lirical_data/ \
  --datasets MME HMS

# Run against a custom dataset directory
python3 run_lirical.py \
  --lirical-jar /path/to/lirical-cli-2.4.0.jar \
  --lirical-data lirical_data/ \
  --data-dir /path/to/your/datasets

# Resume an interrupted run
python3 run_lirical.py \
  --lirical-jar /path/to/lirical-cli-2.4.0.jar \
  --lirical-data lirical_data/ \
  --skip-existing
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--lirical-jar` | Path to LIRICAL JAR. Required. |
| `--lirical-data` | Path to LIRICAL data directory. Required. Default points to `lirical_data/`. |
| `--data-dir` | Directory containing dataset JSON files. Default: `datasets/PhenoBrainBenchmarkDatasets`. |
| `--datasets` | Dataset names to run. Default: all JSON files found in `--data-dir`. |
| `--mindiff` | Minimum number of differential diagnoses to report. Default: `100`. |
| `--java` | Path to Java executable. Default: `java`. |
| `--skip-existing` | Skip cases with existing output (resume mode). Default: off. |

## 6. Implementation
`run_lirical.py` runs in two phases per dataset:

1. **Run:** For each case it invokes LIRICAL's `prioritize` subcommand once (`-p` for the comma-separated HPO terms, `--use-orphanet`, `-f tsv`), writing one `<case_id>.tsv` into `cache/<dataset>/`. Wall-clock time per subprocess call is recorded as `query_time_sec`. With `--skip-existing`, cases whose TSV already exists are skipped (timing is then reported as `skipped`).
2. **Collect:** Each cached TSV is parsed, `ranks` are read from LIRICAL's rank column, and the best (lowest) rank among the confirmed disease IDs is taken as the case result. Rows are assembled into the summary and dataset statistics.

 

## 7. Output Format

Results are written to `output/validation_tools/lirical_benchmarks/<dataset>_summary.tsv`:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Expected disease ID(s), semicolon-separated |
| `rank` | Rank of correct diagnosis (`None` if not found) |
| `matched_id` | Disease ID that matched |
| `score` | Post-test probability reported by LIRICAL |
| `status` | Whether LIRICAL ran successfully |
| `query_time_sec` | Time taken for the query in seconds |

Raw per-case TSV files are cached in `output/validation_tools/lirical_benchmarks/cache/<dataset>/`.

### LIRICAL output columns

| Column | Description |
|--------|-------------|
| `rank` | Disease rank (1 = top candidate) |
| `diseaseName` | Disease name |
| `diseaseCurie` | Disease ID (OMIM or Orphanet CURIE) |
| `pretestprob` | Prior probability |
| `posttestprob` | Posterior probability after phenotype evidence |
| `compositeLR` | Composite likelihood ratio |


## 8. Performance

| Dataset | Cases | Approx. runtime |
|---------|-------|-----------------|
| MME | 40 | ~2 min |
| HMS | 88 | ~4 min |
| LIRICAL | 370 | ~15 min |
| RAMEDIS | 624 | ~25 min |
| PUMCH_L | 988 | ~40 min |
| PUMCH-ADM | 75 | ~3 min |

Runtimes measured on macOS with Java 17. Use `--skip-existing` to safely resume interrupted runs.


## 9. Results
| Dataset |  Found | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Avg. Query Time (s) |
|---------|--------|--------|--------|--------|---------|---------|-------------|
| MME | 40/40 | 0.425 | 0.625 | 0.650 | 0.775 | 0.551 | 9.63 |
| HMS |  88/88 | 0.193 | 0.273 | 0.341 | 0.409 | 0.271 | 12.2 |
| LIRICAL  | 370/370 | 0.449 | 0.600 | 0.630 | 0.665 | 0.536 | 10.83 |
| RAMEDIS  | 375/375 | 0.141 | 0.275 | 0.347 | 0.456 | 0.243 | 11.39 |
| PUMCH_L |  988/988 | 0.891 | 0.212 | 0.255 | 0.367 | 0.182 | 21.17 |
| PUMCH-ADM  | 75/75 | 0.160 | 0.347 | 0.400 | 0.453 | 0.277 | 11.91 |
| GA4GH Phenopackets | 384/384 | 0.451 | 0.591 | 0.625 | 0.680 | 0.537 | 14.53 |
| MyGene2 (5.7.22) | 146/146 | 0.534 | 0.630 | 0.644 | 0.712 | 0.586 | 11.28
| 0.1.27 | 9991/10375 | 0.482 | 0.608 | 0.651| 0.709 | 0.562 | 10.64 |
| test_medical_cases | 200/200 | 0.630 | 0.770 | 0.795 | 0.825 | 0.706 | 35.72 |


## 10. References

> Robinson P.N. et al. *Interpretable clinical genomics with a likelihood ratio paradigm.* Am. J. Hum. Genet. 107, 403–417 (2020). https://doi.org/10.1016/j.ajhg.2020.06.021
