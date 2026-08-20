# TF-IDF Similarity Methods

## Overview

The TF-IDF similarity methods rank rare diseases by weighting HPO terms or clinical text tokens by how **rare** they are across the disease corpus, then computing **cosine similarity** between weighted vectors. Common terms that appear in nearly every disease get low weight; rare, specific terms that appear in only a few diseases get high weight.

Four modes are implemented, each building the patient and disease vectors differently:

- `tfidf_hpo` — HPO ID sets, binary term presence
- `tfidf_text` — free text tokens, word count TF
- `tfidf_hybrid` — patient HPO labels vs disease description text
- `tfidf_hpo_labels` — patient HPO labels vs disease HPO labels

The method is implemented across four files:

```text
config.py        Mode names, thresholds, field names, and output paths
methods.py       IDF computation, vector construction for all four modes
pipeline.py      Pipeline entry point and mode runners
explanation.py   Contribution breakdown, low-IDF flags, IC filter impact
```


## High-Level Method Logic

All four modes follow the same general sequence, with different inputs on each side:

```text
Patient terms or text
        ↓
Build patient TF-IDF vector
        ↓
For each disease:
    Build disease TF-IDF vector
        ↓
    Cosine similarity between patient and disease vectors
        ↓
Rank diseases by score
```


## IDF Weighting

IDF (Inverse Document Frequency) measures how discriminative a term is across the corpus:

```
IDF(term) = log(N / df(term))
```

- `N` = total number of disease profiles
- `df(term)` = number of profiles containing that term

A term in every disease has IDF ≈ 0 and contributes nothing to the score. A term in only one disease has a high IDF and is a strong signal. IDF is computed separately for HPO-mode and text-mode because they operate over different corpora (HPO ID sets vs description text).


## Cosine Similarity

All four modes use the same scorer:

```
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

Both vectors are sparse dicts. The dot product is the sum of products over shared keys. Score is in **[0, 1]**.


## The Four Modes

### HPO Mode (`tfidf_hpo`)

Patient and disease documents are **HPO ID sets**. Binary TF (each term either present or not), so the weight of each term equals its IDF value alone.

```
patient_vec[term] = IDF(term)   for term in patient_propagated_terms ∩ idf_vocab
disease_vec[term] = IDF(term)   for term in disease_propagated_terms ∩ idf_vocab
```

IDF is computed from propagated HPO terms across all disease profiles. Skipped if the patient has no HPO terms.

### Text Mode (`tfidf_text`)

Patient and disease documents are **free text strings**. TF is a real word count, so weight = TF × IDF.

```
patient_vec[token] = count(token in raw_text) × IDF(token)
disease_vec[token] = count(token in merged_description) × IDF(token)
```

IDF is computed from tokenized disease description text. Skipped if the patient has no `raw_text`.

### Hybrid Mode (`tfidf_hybrid`)

The patient document is the **tokenized labels of the patient's HPO terms**; the disease document is the disease's **description text**. This bridges HPO-coded patients to diseases whose descriptions are rich but HPO coding is sparse.

```
patient_vec[token] = count(token in HPO label strings) × IDF(token)
disease_vec[token] = count(token in merged_description) × IDF(token)
```

IC filtering is applied to patient terms before building the label vector: terms below `ic_threshold = 1.5` are removed.

### HPO Labels Mode (`tfidf_hpo_labels`)

Both sides use **tokenized HPO label strings** weighted by text IDF.

```
patient_vec[token] = count(token in patient HPO labels) × IDF(token)
disease_vec[token] = count(token in disease HPO labels) × IDF(token)
```

IC filtering is applied to patient terms. Disease terms are not IC-filtered.


## IC Filtering (Hybrid and HPO Labels modes)

Before building the patient vector in hybrid and hpo_labels modes, terms with IC below the configured threshold (`ic_threshold = 1.5`) are removed. This prevents generic ancestor labels like "All" and "Phenotypic abnormality" from contributing common tokens ("abnormality", "phenotypic") that carry no discriminative signal in the text space.

The filter impact — terms removed and before/after counts — is recorded in the explanation.


## Low-IDF Flagging

The explanation flags matched terms or tokens whose IDF weight is below `LOW_IDF_THRESHOLD = 0.5`. These are very common terms that are unlikely to have driven the score meaningfully. They are surfaced so the user can inspect whether the match is backed by specific or generic overlap.


## Explanation Fields

| Field | Description |
|-------|-------------|
| `summary` | Human-readable one-liner |
| `coverage` | Patient/disease % matched, term counts (HPO mode only) |
| `matched_terms` | Shared HPO terms with IC values (HPO mode only) |
| `tfidf_mode` | Which of the four modes was used |
| `ic_weighted_match_score` | Sum of IC for matched HPO terms (HPO mode) |
| `idf_weighted_score` | Sum of IDF weights for matched tokens (text modes) |
| `contributing_hpo_terms` | Patient HPO terms that contributed matching tokens, with IC and token weights |
| `low_idf_matches` | Matched terms/tokens with IDF below threshold |
| `n_low_idf_matches` | Count of low-IDF matches |
| `vector_norms` | `dot_product`, `patient_norm`, `disease_norm`, `score_check` |
| `ic_filter_impact` | Terms removed by IC threshold, before/after counts (hybrid and labels modes) |


## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LOW_IDF_THRESHOLD` | 0.5 | IDF below this is flagged as common noise |
| `MIN_TOKEN_LENGTH` | 3 | Single and two-character tokens are excluded during tokenization |
| `SPARSE_DISEASE_THRESHOLD` | 5 | Diseases with fewer than 5 tokens are considered sparse |
| `DISEASE_TEXT_FIELD` | `merged_description` | Field used for disease text in text, hybrid, and labels modes |
| `ic_threshold` | 1.5 | IC below this removes patient terms in hybrid and labels modes |


## Run Statistics

The TF-IDF pipeline records per mode:

```text
number of raw patient terms
number of propagated patient terms (or patient text tokens used)
number of diseases scored
number of diseases skipped (empty vector after building)
runtime per mode
```
