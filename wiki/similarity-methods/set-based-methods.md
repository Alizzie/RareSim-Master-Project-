# Set-Based Similarity Methods

## Overview

The set-based similarity methods rank rare diseases by applying classical set and vector operations to patient and disease HPO term sets. Each term is treated as either **present or absent** — there is no ontology structure, no ancestor propagation in the scoring, and no IC weighting in the similarity score itself.

These methods are fast, interpretable, and serve as strong baselines. Their results are easy to explain: the score is determined entirely by how many terms are shared and how large the two sets are.

Four similarity functions are implemented:

- Jaccard
- Dice
- Overlap Coefficient
- Cosine

The method is implemented across four files:

```text
config.py        Method map and output paths
methods.py       Four similarity functions over binary HPO term sets
pipeline.py      Pipeline entry point, term collection, result construction
explanation.py   Formula components, IC-weighted match quality, shared spine fields
```

---

## High-Level Method Logic

The set-based pipeline follows this sequence:

```text
Patient HPO terms (propagated)
        ↓
For each disease:
    Disease HPO terms (propagated)
        ↓
    Apply similarity function to the two sets
        ↓
    Build explanation with formula components and IC match quality
        ↓
Rank diseases by score
```

No IC filtering is applied before scoring. Propagated terms are used on both sides so that ancestor terms are included, which broadens the overlap surface between patient and disease.

---

## Similarity Functions

### Jaccard (`set_jaccard`)

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

The fraction of shared terms out of all terms seen in either set. Penalizes both missed patient terms and extra disease terms equally. Produces values in **[0, 1]**.

- Score = 1.0 → identical term sets
- Score = 0.0 → no shared terms

### Dice (`set_dice`)

```
Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)
```

Similar to Jaccard but gives more weight to the intersection relative to total set size. For any pair of sets, Dice ≥ Jaccard. Produces values in **[0, 1]**.

### Overlap Coefficient (`set_overlap`)

```
Overlap(A, B) = |A ∩ B| / min(|A|, |B|)
```

Measures how well the smaller set is covered by the other. A patient term set that is a complete subset of the disease term set scores 1.0 regardless of how large the disease set is. Useful when disease profiles have many more terms than patient profiles.

### Cosine (`set_cosine`)

```
Cosine(A, B) = |A ∩ B| / (√|A| × √|B|)
```

Treats each set as a binary vector and computes the cosine of the angle between them. Normalizes by both set sizes equally. Produces values in **[0, 1]**.

---

## Formula Components

Each method records its raw numerator and denominator in the explanation so the score can be verified and understood without recomputing:

| Method | Components recorded |
|--------|-------------------|
| `set_jaccard` | `intersection_size`, `union_size` |
| `set_dice` | `intersection_size`, `size_patient`, `size_disease` |
| `set_overlap` | `intersection_size`, `min_size` |
| `set_cosine` | `intersection_size`, `size_patient`, `size_disease` |

---

## IC-Weighted Match Quality

Even though the similarity score is purely set-based, the explanation layer computes an **IC-weighted match score** over the shared terms:

```
IC_match = Σ IC(t)  for t in A ∩ B
```

Two diseases can have the same Jaccard score but different IC match scores if one shares more specific (high-IC) terms. The IC score is surfaced as a quality proxy for the match. The top 5 matched terms by IC are also recorded.

---

## Explanation Fields

| Field | Description |
|-------|-------------|
| `summary` | Human-readable one-liner with matched count, coverage %, and IC quality score |
| `coverage` | Patient % matched, disease % matched, raw term counts |
| `matched_terms` | Shared HPO terms with IC values |
| `unmatched_patient_terms` | Patient terms not found in the disease |
| `formula_components` | Raw numerator/denominator for the score |
| `ic_weighted_match_score` | Sum of IC for all matched terms |
| `top_ic_matched_terms` | Top 5 matched terms by IC descending |

---

## Run Statistics

The set-based pipeline records:

```text
number of raw patient terms
number of propagated patient terms used
number of diseases scored
number of diseases skipped (no terms in profile)
runtime per method
```
