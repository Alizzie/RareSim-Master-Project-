# Dataset Available

## Purpose

This page lists every benchmark dataset currently used for RareSim evaluation, with case counts, phenotype/ground-truth statistics, and sources. Case counts reflect what's present in the evaluation cache after standardization.

**13,040 cases total** across five dataset sources (one of which, the PhenoBrain benchmark, is itself six separate cohorts).

---

## Top-level sources

| Dataset | Cases | Role in evaluation | Source |
|---|---:|---|---|
| PhenoBrain benchmark (6 cohorts) | 1,936 | Benchmark patient cases | [Zenodo](https://zenodo.org/records/10774650) |
| Phenopacket Store 0.1.27 | 10,374 | Benchmark patient cases | [GitHub](https://github.com/monarch-initiative/phenopacket-store/releases) |
| GA4GH Phenopackets | 384 | Benchmark patient cases | [Zenodo](https://zenodo.org/records/3905420) |
| MyGene2 (5.7.22 release) | 146 | Benchmark patient cases | [Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?fileId=6689035&version=3.0) |
| Medical cases | 200 | Raw-text version (for transformer, LLM, and `tfidf_text`), plus a pre-extracted-HPO-terms version with ORPHA ground truth | Ali Khan, S. (2025). *Can LLMs Help with Rare Diseases?* Bachelor's thesis, University of Zurich (UZH). |
| **Total** | **13,040** | | |

---

## PhenoBrain benchmark cohorts

The PhenoBrain benchmark is split into six cohorts, each used independently (as `--dataset MME`, `--dataset HMS`, etc.):

| Cohort | Cases | HPO/case | GT/case | Multi-GT | Namespaces | Dup. |
|---|---:|---:|---:|---:|---|---:|
| MME | 40 | 12.2 ± 6.2 | 2.05 | 97.5% | O / Or / C | 0 |
| HMS | 88 | 19.4 ± 10.6 | 3.19 | 76.1% | O / Or / C | 3 |
| LIRICAL | 370 | 14.3 ± 13.9 | 1.63 | 61.4% | O / Or / C | 0 |
| PUMCH_L | 988 | 32.1 ± 16.1 | 9.33 | 92.5% | O / Or / C | 0 |
| PUMCH-ADM | 75 | 18.6 ± 10.4 | 7.35 | 100.0% | O / Or / C | 0 |
| RAMEDIS | 375 | 10.5 ± 6.7 | 2.74 | 100.0% | Or / O / C | 24 |

---

## Other dataset sources

| Dataset | Cases | HPO/case | GT/case | Multi-GT | Namespaces | Dup. |
|---|---:|---:|---:|---:|---|---:|
| Phenopacket Store 0.1.27 | 10,374 | 8.9 ± 6.5 | 1.71 | 70.8% | O / Or | 1,374 |
| GA4GH Phenopackets | 384 | 13.8 ± 13.8 | 1.99 | 99.5% | O / Or | 4 |
| MyGene2 (5.7.22) | 146 | 7.7 ± 6.5 | 2.26 | 95.9% | Or / O | 19 |
| Medical cases (extracted version) | 200 | 57.3 ± 24.2 | 1.00 | 0.0% | Or | 0 |

**Column key** (same for both tables above): `HPO/case` and `GT/case` are the mean ± SD HPO term count and mean ground-truth cardinality per case; `Multi-GT` is the share of cases with more than one ground-truth disease; `Namespaces` lists the disease ID systems present (O = OMIM, Or = ORPHA, C = CCRD), most common first; `Dup.` is the number of cases in that dataset sharing an identical (HPO term set, ground truth) fingerprint with another case in the same dataset.

---

## Reading the statistics

**Phenotype density varies by close to an order of magnitude across sources** — MyGene2 cases average 7.7 HPO terms while Medical cases (extracted version) averages 57.3. A method's apparent strength or weakness on one dataset doesn't necessarily generalize to another with a very different term-count profile, so cross-dataset comparisons should account for this.

**Ground-truth cardinality is not uniform either.** Medical cases has exactly one ground-truth disease per case throughout (0% Multi-GT), while MME, HMS, PUMCH-ADM, and RAMEDIS have more than one ground-truth disease in *every* case, and PUMCH_L averages over nine. This is why the evaluator's rank-based metrics are defined per case as the best rank across all ground-truth diseases (`rank_i = min` over the ground-truth set `G_i`) rather than assuming a single correct answer — see [evaluator-and-metrics.md](evaluator-and-metrics.md). A large share of cases genuinely have more than one acceptable answer.

**Phenopacket Store version 0.1.27 contains 1,374 duplicate cases among 10,374 records, corresponding to approximately 13% of the dataset.** This is compatible with a documented limitation of the source data rather than being introduced by RareSim preprocessing. Danis et al. (2025) note that the same individual may be described in multiple publications under different identifiers and that duplicate detection is not handled by the Phenopacket Schema itself. RareSim does not perform cross-case deduplication during standardization. Consequently, duplicate records already present in the source dataset may remain in the standardized benchmark.

---

## Raw-text availability

All benchmark datasets except **Medical cases** have 0% raw-text availability in the evaluation cache — they are distributed as pre-structured HPO-term benchmarks, not raw clinical narratives, so only the [standard HPO-term format](dataset-format.md#standard-hpo-term-format) applies to them.

Medical cases is the exception, and exists in two representations of the same 200 patients:

- **HPO-based representation.** Phenotype terms were extracted from the underlying clinical text as a one-time offline preprocessing step when the benchmark was built. Only the extracted HPO terms and their ORPHA ground truth are provided to phenotype-based retrieval methods. This representation exercises RareSim's *ranking* methods on already-extracted phenotype profiles — it does not itself re-run or benchmark RareSim's HPO extraction pipeline, since that extraction happened upstream and independently.
- **Raw-text representation.** Preserves the original clinical descriptions together with the same ORPHA ground-truth identifiers, in the [raw-text format](dataset-format.md#raw-text-format). This is what `run_transformer_text.py`, `run_llm_text.py`, and `run_tfidf_text.py` consume: the transformer and LLM methods use the clinical description directly, while `tfidf_text` compares the patient narrative against disease text representations. The two representations contain the same 200 cases and differ only in what patient information is supplied to the method.

---

## Contributing a new dataset

See [dataset-adding.md](dataset-adding.md) for how to prepare and register a new dataset, and [dataset-format.md](dataset-format.md) for the JSON schemas above. If you add or update a dataset, please update the tables on this page (case count, HPO/case, GT/case, Multi-GT, namespaces, duplicate count, and source) so this remains the single reference point for what's available.
