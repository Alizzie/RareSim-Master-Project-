# Evaluation Cache Format

## Purpose

Each evaluation runner writes per-case results into cache files.

The cache allows different methods to be run independently and evaluated together later.

Cache directory:

```text
outputs/evaluation/<DATASET>/cache/
```

Example:

```text
outputs/evaluation/MME/cache/
```

Each case is saved as:

```text
case_0000.json
case_0001.json
case_0002.json
...
```

---

## Cache path

Function:

```python
cache_path_for(cache_dir, index)
```

Example:

```python
cache_path_for(cache_dir, 0)
```

returns:

```text
outputs/evaluation/MME/cache/case_0000.json
```

---

## Cache file structure

Example:

```json
{
  "case_index": 0,
  "hpo_terms": ["HP:0001263", "HP:0001250"],
  "ground_truth": ["ORPHA:123"],
  "total_elapsed_seconds": 12.345,
  "method_elapsed_seconds": {
    "set_jaccard": 0.021,
    "set_cosine": 0.020
  },
  "methods_run": [
    "set_cosine",
    "set_jaccard"
  ],
  "results": {
    "set_jaccard": [
      {
        "disease_id": "ORPHA:123",
        "label": "Disease name",
        "score": 0.87,
        "rank": 1
      }
    ]
  }
}
```

---

## Required fields

### `case_index`

Integer case index.

### `hpo_terms`

The patient HPO terms used for the case.

### `ground_truth`

The confirmed disease IDs for the case.

A case may have multiple ground-truth IDs.

### `total_elapsed_seconds`

Accumulated runtime for this case.

When multiple runners add results to the same case file, `save_cache()` adds the new runtime to the existing total.

### `method_elapsed_seconds`

Runtime per method.

Example:

```json
"method_elapsed_seconds": {
  "set_jaccard": 0.021,
  "tfidf": 0.054
}
```

### `methods_run`

Sorted list of methods available in the cache file.

Used by resume logic.

### `results`

Ranked disease results per method.

Example:

```json
"results": {
  "set_jaccard": [
    {
      "disease_id": "ORPHA:123",
      "label": "Disease name",
      "score": 0.87,
      "rank": 1
    }
  ]
}
```

---

## Result fields

Each result should contain at least:

```text
disease_id or canonical_disease_id or ordo_id
rank
score
label
```

Preferred format:

```json
{
  "disease_id": "ORPHA:123",
  "label": "Example disease",
  "score": 0.87,
  "rank": 1
}
```

The evaluator can extract disease IDs from:

```text
disease_id
canonical_disease_id
ordo_id
```

---

## Cache merging

Function:

```python
save_cache(...)
```

The save function:

```text
1. Loads an existing case cache if it exists.
2. Merges new method results into existing results.
3. Merges new method timing into existing timing.
4. Updates methods_run.
5. Writes the file back to disk.
```

This means these commands can be run independently:

```bash
python scripts/evaluation/run_set_based.py --test-set data/datasets/phenobrain_testdata/MME.json
python scripts/evaluation/run_tfidf.py --test-set data/datasets/phenobrain_testdata/MME.json
python scripts/evaluation/run_semantic.py --test-set data/datasets/phenobrain_testdata/MME.json
```

All runners write into:

```text
outputs/evaluation/MME/cache/
```

and each case file accumulates results from all methods.

---

## Resume behavior

Function:

```python
methods_already_cached(cache_file, required_methods)
```

Returns `True` if every required method is already present in the cache file.

Default behavior:

```text
resume = True
```

Disable resume:

```bash
python scripts/evaluation/run_set_based.py \
    --test-set data/datasets/phenobrain_testdata/MME.json \
    --no-resume
```

Use `--no-resume` when:

```text
old cache should be overwritten
```

---

## Error files

If a case fails, the runner writes an error file next to the cache files.

Example:

```text
case_0003.error
```

The file contains:

```text
ExceptionType: error message
```

This allows the batch process to continue even if one case fails.