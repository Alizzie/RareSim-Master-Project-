# Dataset Format

## Purpose

This page documents the JSON schemas that batch runners accept for a benchmark test set, and which runners accept which schema. There are three shapes, derived directly from the loader code in `scripts/evaluation/`.

```text
Standard (HPO-term)   -> run_set_based.py, run_semantic.py, run_tfidf.py,
                          run_hpo2vec.py, run_autoencoder.py,
                          run_transformer.py, run_llm.py
Raw-text               -> run_tfidf_text.py, run_transformer_text.py,
                          run_llm_text.py
Negative-aware         -> run_set_jaccard_penalized.py
```


## Standard (HPO-term) format

Loaded by `load_test_cases(path)` in `_batch_utils.py`.

```json
[
  [
    ["HP:0001263", "HP:0001250"],
    ["ORPHA:123"]
  ],
  [
    ["HP:0004322", "HP:0000252"],
    ["OMIM:123456", "ORPHA:999"]
  ]
]
```

The root is a JSON array. Each element is a 2-element array:

```text
[0]  hpo_terms    list[str]   patient phenotype terms (HPO IDs, e.g. "HP:0001263")
[1]  ground_truth list[str]   expected disease IDs (e.g. "ORPHA:123", "OMIM:123456")
```

A case may have more than one ground-truth ID — this is used both for genuinely multi-diagnosis cases and for listing cross-references (an OMIM and an ORPHA code for the same disease). The evaluator treats alias-equivalent ground-truth IDs as one relevant disease for NDCG purposes (see [evaluator-and-metrics.md](evaluator-and-metrics.md)).

There is no `id` field in this format — cases are identified purely by their position in the array, and cached as `case_{index:04d}`.


## Raw-text format

Loaded independently by `run_tfidf_text.py`, `run_transformer_text.py`, and `run_llm_text.py` (each has its own local copy of the same loading logic). Two shapes are accepted.

Of the datasets currently in use, only "Medical cases" is distributed in this format (alongside a separately-extracted HPO-term version of the same 200 patients) — see [dataset-available.md](dataset-available.md#raw-text-availability). Every other benchmark dataset is HPO-term only.

### Shape 1: list of case objects

```json
[
  {
    "id": "case_0000",
    "raw_text": "45-year-old patient presenting with recurrent seizures, developmental delay, and distinctive facial features...",
    "disease_codes": ["ORPHA:58"]
  }
]
```

Fields:

```text
id             optional, string. Defaults to "case_{index:04d}" if missing or blank.
raw_text       required. The clinical text. Aliases: "text", "clinical_text", "description".
disease_codes  required. A string or a list of strings.
               Aliases: "ground_truth", "disease_id", "disease_code", "orpha_code".
```

Only the first alias present is used per field (e.g. if both `raw_text` and `text` exist, `raw_text` wins). Ground truth is normalized to a sorted, de-duplicated list of strings; bare numeric ground-truth values (e.g. `"58"`) are normalized to `ORPHA:58`.

### Shape 2: disease-to-text mapping

```json
{
  "58": "Patient clinical text...",
  "ORPHA:61": "Another patient clinical text..."
}
```

The root is a JSON object mapping a disease identifier (bare numeric or already-prefixed) to the clinical text for one case. Each key/value pair becomes one case, with `ground_truth = [normalized_key]` and `case_id = "case_{index:04d}"` (index = position in iteration order).

Both shapes require every text value to be a non-empty string after stripping whitespace, or the loader raises an error naming the offending case index.


## Negative-aware format

Loaded by `load_negative_aware_test_cases(path)` in `run_set_jaccard_penalized.py` only.

```json
[
  {
    "hpo_terms": ["HP:0001263", "HP:0001250"],
    "excluded_hpo_terms": ["HP:0000750"],
    "disease_codes": ["ORPHA:123", "OMIM:123456"]
  }
]
```

Fields:

```text
hpo_terms           required, non-empty list[str]. Positive (present) phenotype terms.
excluded_hpo_terms  optional, list[str]. May be empty. Terms explicitly noted absent/negated.
disease_codes       required, non-empty list[str]. Ground-truth disease IDs.
```

All three fields are normalized (deduplicated and stripped) the same way. This format is designed to layer onto an existing HPO-term dataset — see `--cache-name` and the `_with_excluded` filename convention in [batch-runners-and-shared-utilities.md](batch-runners-and-shared-utilities.md) and [workflow-overview.md](workflow-overview.md).


## Validation behavior common to all three formats

All loaders read the whole file with `json.loads(...)`, then validate case-by-case, raising `ValueError` (with the offending case index in the message) on the first problem found — they do not silently skip or coerce malformed cases. Practical implications when preparing a dataset:

```text
Every text/HPO-term field must be non-empty after stripping whitespace.
Ground-truth / disease-code fields must contain only strings, not
    numbers, nulls, or nested objects.
The root value's type (list vs. object) determines which loader path
    is taken automatically — you don't declare the shape explicitly.
```

For guidance on adding a brand-new dataset file to the benchmarking suite, see [dataset-adding.md](dataset-adding.md).
