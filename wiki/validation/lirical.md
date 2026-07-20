# LIRICAL

LIRICAL (LIkelihood Ratio Interpretation of Clinical AbnormaLities) is a phenotype-driven disease prioritisation tool that uses a likelihood ratio framework to rank candidate diagnoses given a set of HPO terms.

- **Repository:** [TheJacksonLaboratory/LIRICAL](https://github.com/TheJacksonLaboratory/LIRICAL)
- **Documentation:** [LIRICAL Setup Guide](https://thejacksonlaboratory.github.io/LIRICAL/stable/setup.html)
- **Version:** `2.4.0`
- **Tested on:** macOS, May 2026

> **Important:** The YAML input format changed between versions. In v2.4.0, the `yaml` subcommand does not work correctly — the runner uses the `prioritize` subcommand with `-p` flags instead.

---

## Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Java | 17+ | Verify with `java -version` |
| RAM | 8 GB+ | Recommended |
| Disk | ~500 MB | For LIRICAL data directory |

---

## 1. Install LIRICAL

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

---

## 2. Data Files

LIRICAL requires a data directory containing the HPO ontology and gene/transcript annotation files. Run the built-in download command once:

```bash
cd lirical-cli-2.4.0
lirical download
```

> This downloads ~500 MB and only needs to be done once. For phenotype-only mode (no VCF), Exomiser variant databases are not needed. See the [official setup guide](https://thejacksonlaboratory.github.io/LIRICAL/stable/setup.html) for details.
> If you want the data in a different location, you need to pass that path explicitly when running the benchmark via --lirical-data.

---

## 3. Verify Installation

Test with a single case using the `prioritize` subcommand:

```bash
java -jar /path/to/lirical-cli-2.4.0.jar prioritize \
  -p HP:0002321,HP:0000365,HP:0000360,HP:0009589,HP:0002858 \
  -n HP:0009736 \
  -d lirical_data/
```

If it outputs a ranked disease table, the setup is correct.

---

## 4. Running the Benchmark

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

---

## 5. Output Format

Results are written to `lirical_benchmarks/<dataset>_summary.tsv`:

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

Raw per-case TSV files are cached in `lirical_benchmarks/cache/<dataset>/`.

### LIRICAL output columns

| Column | Description |
|--------|-------------|
| `rank` | Disease rank (1 = top candidate) |
| `diseaseName` | Disease name |
| `diseaseCurie` | Disease ID (OMIM or Orphanet CURIE) |
| `pretestprob` | Prior probability |
| `posttestprob` | Posterior probability after phenotype evidence |
| `compositeLR` | Composite likelihood ratio |

---

## Performance

| Dataset | Cases | Approx. runtime |
|---------|-------|-----------------|
| MME | 40 | ~2 min |
| HMS | 88 | ~4 min |
| LIRICAL | 370 | ~15 min |
| RAMEDIS | 624 | ~25 min |
| PUMCH_L | 988 | ~40 min |
| PUMCH-ADM | 75 | ~3 min |

Runtimes measured on macOS with Java 17. Use `--skip-existing` to safely resume interrupted runs.

---

## References

> Robinson P.N. et al. *Interpretable clinical genomics with a likelihood ratio paradigm.* Am. J. Hum. Genet. 107, 403–417 (2020). https://doi.org/10.1016/j.ajhg.2020.06.021
