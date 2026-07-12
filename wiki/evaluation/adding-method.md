# Adding a New Evaluation Method

## Purpose

This page explains how to add a new method to the RareSim evaluation workflow.

The evaluator automatically detects methods from cache files. Therefore, a new method does not need evaluator changes if it writes results in the expected format.

---

## Where a new method fits

A new method usually needs two parts:

```text
1. Method implementation
       raresim/similarity_methods/<method_name>/

2. Evaluation runner
       scripts/evaluation/run_<method_name>.py
```

The method implementation computes rankings.

The evaluation runner runs the method over all test cases and writes results into the shared evaluation cache format.

---

## Required cache contract

Each case cache must include the method under:

```text
case["results"][METHOD_NAME]
```

Each result should contain at least:

```text
disease_id or canonical_disease_id or ordo_id
rank
score
label
```

The method runtime should be saved under:

```text
case["method_elapsed_seconds"][METHOD_NAME]
```

The method name should be included in:

```text
case["methods_run"]
```

Minimal result example:

```json
{
  "disease_id": "ORPHA:123",
  "label": "Example disease",
  "score": 0.87,
  "rank": 1
}
```

---

## Step 1: Implement the method

Add the method under:

```text
raresim/similarity_methods/<method_name>/
```

Recommended structure:

```text
raresim/similarity_methods/<method_name>/
    __init__.py
    pipeline.py
    methods.py
    config.py
```

The method pipeline should return ranked disease results.

Recommended output:

```text
list[SimilarityResult]
```

or:

```text
list[dict]
```

If using `SimilarityResult`, make sure it can be serialized with `.to_dict()`.

---

## Step 2: Create a batch runner

Create:

```text
scripts/evaluation/run_<method_name>.py
```

Use existing runners as templates.

For CPU-style HPO methods:

```text
run_set_based.py
run_semantic.py
run_tfidf.py
```

For embedding/model methods:

```text
run_hpo2vec.py
run_autoencoder.py
run_transformer.py
```

For generative model methods:

```text
run_llm.py
```

---

## Step 3: Use common batch utilities

Import shared helpers:

```python
from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    add_common_args,
    build_patient,
    cache_path_for,
    load_test_cases,
    methods_already_cached,
    print_case,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
    serialize_results,
)
```
---

## Step 4: Define a stable method name

Example:

```python
METHOD_NAME = "my_new_method"
METHOD_NAMES = [METHOD_NAME]
```

This string becomes the key in:

```text
results
method_elapsed_seconds
methods_run
```

Do not rename it after generating caches unless old caches are regenerated or migrated.

---

## Step 5: Load test cases and shared context

Typical setup:

```python
cases = load_test_cases(test_set_path)

dummy = PatientProfile("batch_init", "", set(), set())
ctx = AppContext.load(dummy, use_canonical_profiles=True)
ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
```

Use canonical profiles by default unless the method specifically needs expanded alias profiles.

---

## Step 6: Build patient input

Build the patient profile with the shared helper:

```python
patient = build_patient(index, hpo_terms, ancestor_sets)

The evaluation patient ID should follow:

```python
f"eval_case_{index:04d}"
```

---

## Step 7: Run the method and produce ranked results

Each result needs:

```text
disease_id
label
score
rank
```

If the method uses another disease ID field, the evaluator supports:

```text
canonical_disease_id
ordo_id
```

but `disease_id` is preferred for consistency.

---

## Step 8: Save results to cache

Use:

```python
save_cache(
    cache_file,
    index,
    hpo_terms,
    ground_truth,
    {METHOD_NAME: results},
    {METHOD_NAME: elapsed},
    elapsed,
)
```

If the method returns `SimilarityResult` objects, use:

```python
serialize_results(results)
```

if the output is grouped by method.

This merges the new method results with existing results in the same case file.

---

## Step 9: Run the evaluator

After the new method has written results for the dataset:

```bash
python scripts/evaluation/evaluator.py --dataset MME
```

The evaluator automatically detects the new method from the cache files.

No evaluator changes are needed if the cache format is correct.

---

## Minimal new runner template

```python
"""Batch runner for RareSim my_method similarity."""

import argparse
from pathlib import Path

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.timer import Timer

from _batch_utils import (
    EVALUATION_DIR,
    add_common_args,
    build_patient,
    cache_path_for,
    load_test_cases,
    methods_already_cached,
    print_case,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
    serialize_results,
)

from raresim.similarity_methods.my_method.pipeline import run as run_my_method

METHOD_NAME = "my_method"
METHOD_NAMES = [METHOD_NAME]


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run my_method on every test case."""
    if config is None:
        config = PipelineConfig(
            use_propagated_terms=True,
            use_canonical_profiles=True,
        )

    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(METHOD_NAME, test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    print("Loading shared context...")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, config.use_canonical_profiles)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print("  Ready.\n")

    skipped = 0
    processed = 0
    failed = 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, METHOD_NAMES):
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print_case(index, total, hpo_terms, ground_truth)

        try:
            case_timer = Timer(METHOD_NAME).start()

            results = run_my_method(
                patient,
                METHOD_NAMES,
                config,
                ctx,
            )

            elapsed = round(case_timer.stop(), 3)
            total_time += elapsed

            save_cache(
                cache_file,
                index,
                hpo_terms,
                ground_truth,
                serialize_results(results),
                {METHOD_NAME: elapsed},
                elapsed,
            )

            processed += 1
            print_case_ok(elapsed, total_time, processed, total - index - 1)

        except Exception as error:
            failed += 1
            print_case_err(error)
            (cache_dir / f"case_{index:04d}.error").write_text(
                f"{type(error).__name__}: {error}",
                encoding="utf-8",
            )

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RareSim my_method batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    config = PipelineConfig(
        top_k=args.top_k,
        use_propagated_terms=True,
        use_canonical_profiles=True,
    )

    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        config=config,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
```

---

## New method checklist

Before running the evaluator, check:

```text
The new runner creates cache files under:
    outputs/evaluation/<DATASET>/cache/

Each case file contains:
    results[MY_METHOD]

Each result contains:
    disease_id or canonical_disease_id or ordo_id
    rank
    score
    label

Each case file contains timing:
    method_elapsed_seconds[MY_METHOD]

Each case file lists the method:
    methods_run includes MY_METHOD

The method name is stable and consistent across files.

The evaluator can extract the disease ID.

The method has been run on enough cases to compare fairly.

After running:
    python scripts/evaluation/evaluator.py --dataset <DATASET>
```

---

## Common mistakes

### Method name mismatch

If the runner saves results under one method name but checks resume using another, caching and evaluation become inconsistent.

Use one stable name:

```python
METHOD_NAME = "my_method"
METHOD_NAMES = [METHOD_NAME]
```

### Missing rank

The evaluator needs `rank`.

Every result should have:

```json
"rank": 1
```

### Missing disease ID

The evaluator must be able to extract one of:

```text
disease_id
canonical_disease_id
ordo_id
```

### Missing timing

Metrics still work, but average runtime will be unavailable.

### Changing result schema

Prefer the standard format:

```json
{
  "disease_id": "...",
  "label": "...",
  "score": 0.0,
  "rank": 1
}
```
