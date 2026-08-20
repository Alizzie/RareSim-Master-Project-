# Evaluation Cache Format

## Purpose

Each evaluation runner writes per-case results into cache files.

The cache allows different methods (and different runners, run at different times) to be run independently and evaluated together later — every batch runner merges its results into the same per-case file rather than overwriting it.

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

`<DATASET>` is normally the test-set filename stem, but the `_text` runners and `run_set_jaccard_penalized.py` accept `--cache-name` to target an existing dataset's cache explicitly (see [batch-runners-and-shared-utilities.md](batch-runners-and-shared-utilities.md)).


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


## Cache file structure (HPO-term cases)

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

### Raw-text cases

`run_tfidf_text.py`, `run_llm_text.py`, and `run_transformer_text.py` write the same base structure, but with `hpo_terms` set to `[]` and two extra fields merged in after `save_cache()` runs:

```json
{
  "case_index": 0,
  "hpo_terms": [],
  "case_id": "case_0000",
  "raw_text": "45-year-old patient presenting with...",
  "ground_truth": ["ORPHA:123"],
  "...": "same result/timing fields as above"
}
```

`case_id` comes from the test-set entry's `id` field (or defaults to `case_{index:04d}`). Each of these runners also validates, before writing, that an existing cache file's `ground_truth` and `raw_text` match the current test-set entry for that case index — a mismatch raises an error rather than silently overwriting, to prevent two different datasets from being merged into the same cache by accident.

### Negative-aware cases

`run_set_jaccard_penalized.py` adds one more field, `excluded_hpo_terms`:

```json
{
  "case_index": 0,
  "hpo_terms": ["HP:0001263", "HP:0001250"],
  "excluded_hpo_terms": ["HP:0000750"],
  "ground_truth": ["ORPHA:123"],
  "...": "same result/timing fields as above"
}
```

If the case already exists in the cache (written earlier by another runner such as `run_set_based.py`), this runner preserves the existing `hpo_terms` and `ground_truth` rather than overwriting them with the negative-aware test set's values, and only warns (without failing) if the ground truth differs.


## Required fields

### `case_index`

Integer case index.

### `hpo_terms`

The patient HPO terms used for the case. Empty (`[]`) for raw-text cases.

### `ground_truth`

The confirmed disease IDs for the case. A case may have multiple ground-truth IDs, including IDs that are aliases of the same underlying disease (see [evaluator-and-metrics.md](evaluator-and-metrics.md) for how the evaluator collapses these for NDCG).

### `total_elapsed_seconds`

Accumulated runtime for this case. When multiple runners add results to the same case file, `save_cache()` adds the new runtime to the existing total rather than replacing it.

### `method_elapsed_seconds`

Runtime per method:

```json
"method_elapsed_seconds": {
  "set_jaccard": 0.021,
  "tfidf": 0.054
}
```

Some runners (e.g. `run_hpo2vec.py`) record one combined timer under a single pipeline name even when multiple method variants are produced in `results` — see the runner's own documentation for which granularity it uses.

### `methods_run`

List of methods available in the cache file. Used by resume logic (`methods_already_cached`).

### `results`

Ranked disease results per method:

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

### `case_id`, `raw_text` (raw-text cases only)

Written by the `_text` runners after `save_cache()`, so they survive subsequent merges from other `_text` runners writing into the same cache file.

### `excluded_hpo_terms` (negative-aware cases only)

Written by `run_set_jaccard_penalized.py` after `save_cache()`.


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

The evaluator extracts disease IDs from, in order: `disease_id`, `canonical_disease_id`, `ordo_id`. See `get_disease_id_from_result()` in [evaluator-and-metrics.md](evaluator-and-metrics.md).


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
5. Adds the new elapsed time to the existing total_elapsed_seconds.
6. Writes the file back to disk.
```

This means these commands can be run independently, in any order, and their results accumulate into the same cache:

```bash
python scripts/evaluation/run_set_based.py --test-set data/datasets/phenobrain_testdata/MME.json
python scripts/evaluation/run_tfidf.py --test-set data/datasets/phenobrain_testdata/MME.json
python scripts/evaluation/run_semantic.py --test-set data/datasets/phenobrain_testdata/MME.json
```

All three write into:

```text
outputs/evaluation/MME/cache/
```

and each case file accumulates results from all methods.

The raw-text and negative-aware runners layer additional metadata (`case_id`/`raw_text` or `excluded_hpo_terms`) on top of this same merge behavior — see above.


## Resume behavior

Function:

```python
methods_already_cached(cache_file, required_methods)
```

Returns `True` if every required method is already present in the cache file's `methods_run`.

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

Use `--no-resume` when old cached results for that method should be recomputed and overwritten (e.g. after a bug fix in the method implementation).


## Error files

If a case fails, the runner writes an error file next to the cache files instead of a case cache, so the case can be diagnosed and retried:

```text
case_0003.error
```

The file contains:

```text
ExceptionType: error message
```

This allows the batch process to continue processing the remaining cases even if one case fails. A failed case is not marked in `methods_run`, so re-running the same command later will retry it (resume only skips cases where the method already succeeded).
