# Semantic Similarity Methods

## Overview

The semantic similarity methods rank rare diseases by comparing patient and disease HPO term sets using the **ontology structure** and **Information Content (IC)** values. Instead of counting shared terms directly, they measure how *specific* the shared ancestors between terms are.

Two terms that share a rare, specific ancestor score higher than two terms that only share a generic one like "Abnormality of the nervous system". This means the methods reward specificity — a patient with a precise, rare phenotype matching a disease's precise phenotype is scored higher than a vague match on generic terms.

Three pairwise similarity functions are implemented, all using the same **Best Match Average (BMA)** aggregation strategy to extend pairwise term scores to full term set comparisons:

- Resnik BMA
- Lin BMA
- Jiang-Conrath BMA

The method is implemented across four files:

```text
config.py        Method map, constants, weak match threshold, and output paths
methods.py       Pairwise IC similarity functions, MICA computation, BMA aggregation
pipeline.py      Pipeline entry point, IC filtering, bidirectional BMA scoring
explanation.py   BMA directions, semantic clusters, weak matches, IC filter impact
```


## High-Level Method Logic

The semantic pipeline follows this sequence:

```text
Patient HPO terms (propagated)
        ↓
IC filtering — remove terms below ic_threshold
        ↓
For each disease:
    IC filtering on disease terms
        ↓
    Compute pairwise similarity: patient terms × disease terms
        ↓
    BMA patient→disease (p2d)
        ↓
    BMA disease→patient (d2p)
        ↓
    Final score = 0.5 × (p2d + d2p)
        ↓
Rank diseases by score
```


## Information Content (IC)

IC measures how specific a term is across the disease corpus:

```
IC(term) = -log(df(term) / N)
         = log(N / df(term))
```

- `N` = total number of disease profiles
- `df(term)` = number of profiles containing that term

A term appearing in every disease has IC ≈ 0 and carries no discriminative signal. A term appearing in only one disease has a high IC and is a strong diagnostic signal. IC values are precomputed once and shared across all semantic methods.


## Most Informative Common Ancestor (MICA)

For any two HPO terms, the MICA is the shared ancestor with the highest IC:

```
MICA(a, b) = argmax IC(t)  for t in ancestors(a) ∩ ancestors(b)
```

Ancestor sets are computed inclusively, meaning each term is its own ancestor. MICA computation results are cached at module level keyed on `(term_a, term_b)`. Since ancestor sets and IC values are fixed for an entire batch run, the same pair always produces the same result. Call `clear_mica_cache()` between patients if memory becomes a concern.


## Pairwise Similarity Functions

### Resnik (`semantic_resnik_bma`)

```
Resnik(a, b) = IC(MICA(a, b))
```

Measures similarity purely by the specificity of the shared ancestor. Ignores how far each term is from the MICA. Scores are unbounded above — higher IC means more specific means higher score.

### Lin (`semantic_lin_bma`)

```
Lin(a, b) = 2 × IC(MICA) / (IC(a) + IC(b))
```

Normalizes Resnik by the specificity of both terms. Produces values in **[0, 1]**. Returns 0 if either term has IC = 0.

### Jiang-Conrath (`semantic_jiang_conrath_bma`)

```
JC_distance(a, b) = IC(a) + IC(b) − 2 × IC(MICA)
JC_similarity(a, b) = 1 / (1 + distance)
```

Distance-based measure converted to similarity. Bounded in **(0, 1]**. Returns 0.0 if no common ancestor exists.


## Best Match Average (BMA)

All three methods use BMA aggregation to compare two *sets* of terms:

```
For each source term:
    best_score = max pairwise similarity against all target terms

BMA(source_set, target_set) = mean(best_scores)
```

The pipeline computes BMA in both directions and averages them:

```
score = 0.5 × (p2d_avg + d2p_avg)
```

This bidirectional averaging prevents a disease with many terms from always outscoring one with few, because the disease-to-patient direction penalizes diseases whose terms are not explained by the patient.


## IC Filtering

Before scoring, terms with IC below the configured threshold (`ic_threshold = 1.5`) are removed from both patient and disease term sets. Generic ancestors like "All" (IC ≈ 0.4) and "Phenotypic abnormality" (IC ≈ 0.41) are filtered out because they carry no discriminative signal and would bias the MICA toward generic shared ancestors.

The filter impact — how many terms were removed and which ones — is recorded in the explanation.


## Semantic Clusters

The explanation layer groups patient-to-disease BMA matches by their shared MICA. When multiple patient terms all route through the same MICA, they form a semantic cluster:

```
"5 patient terms cluster around Cerebellar ataxia"
```

This gives a higher-level view of why the score is high: instead of listing individual term matches, clusters reveal the shared semantic concept driving the similarity. Only clusters with at least 2 patient terms are shown.


## Weak Matches

Patient terms whose best BMA partner scored below `WEAK_MATCH_THRESHOLD = 0.3` are flagged as weak matches. These are the terms that reduce the overall score — phenotypes the patient has that the disease does not explain well. Clinically, they may represent atypical features or suggest an alternative diagnosis.

Weak matches are sorted by IC descending so the most notable unexplained features appear first.


## BMA Asymmetry

The explanation records both BMA directions and computes an asymmetry metric:

```
asymmetry = |p2d_avg − d2p_avg|
```

The interpretation is classified as:

| Label | Meaning |
|-------|---------|
| `symmetric` | Both directions agree within 0.15 |
| `patient_better_covered` | Patient terms match disease well; disease has many extra terms |
| `disease_better_covered` | Disease terms match patient well; patient has unexplained terms |


## Explanation Fields

| Field | Description |
|-------|-------------|
| `summary` | Human-readable one-liner with coverage, BMA scores, top cluster, and weak match count |
| `coverage` | Patient % matched, disease % matched, term counts |
| `matched_terms` | HPO terms shared between patient and disease (with IC values) |
| `unmatched_patient_terms` | Patient terms not found in the disease |
| `bma_variant` | Short name of the pairwise method (resnik, lin, jiang_conrath) |
| `bma_directions` | `patient_to_disease_avg`, `disease_to_patient_avg`, `asymmetry`, `asymmetry_interpretation` |
| `semantic_clusters` | Patient terms grouped by shared MICA, sorted by cluster average score |
| `weak_patient_matches` | Patient terms with BMA score below 0.3, sorted by IC |
| `ic_filter_impact` | Terms removed by IC threshold filtering, before/after counts |


## Run Statistics

The semantic pipeline records:

```text
number of raw patient terms
number of propagated patient terms before IC filtering
number of patient terms used after IC filtering
number of diseases scored
number of diseases skipped (no terms after filtering)
runtime per method
```
