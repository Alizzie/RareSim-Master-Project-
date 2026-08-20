# Dataset Adding

## Purpose

This page covers how to add a new patient test-set file to the benchmarking suite so it can be run through the batch runners and evaluator.


## 1. Choose the right format

Pick the schema that matches the data you have — see [dataset-format.md](dataset-format.md) for full details:

```text
Have patient HPO terms + confirmed diagnosis?
    -> Standard (HPO-term) format

Have raw clinical text (no coded HPO terms) + confirmed diagnosis?
    -> Raw-text format

Have HPO terms plus explicitly negated/excluded findings, and want
to layer a penalized-method pass onto an existing HPO-term dataset?
    -> Negative-aware format
```

The format determines which batch runners can consume the file — the standard and raw-text formats are mutually exclusive per file (a runner picks its loader based on the file's shape and field names), so if you want both HPO-term and raw-text evaluation for the same patients, prepare two files.


## 2. Place the file

There isn't a single required directory — runners take an explicit `--test-set` path — but the existing convention in the command examples throughout this wiki is:

```text
data/datasets/phenobrain_testdata/<NAME>.json      standard HPO-term sets
data/datasets/free_text/<NAME>.json                raw-text sets
data/datasets/phenopackets/standardized_to_json<NAME>_with_excluded.json   negative-aware sets
```


## 3. Name the dataset

The dataset name used for the evaluation cache directory and for `evaluator.py --dataset <NAME>` is the test-set filename stem (`test_set_path.stem`) by default:

```text
data/datasets/phenobrain_testdata/MME.json  ->  outputs/evaluation/MME/
```

Raw-text and negative-aware runners accept `--cache-name` to override this — useful when you want a raw-text file's results to merge into an existing HPO-term dataset's cache, or vice versa, rather than getting a cache directory of its own.


## 4. Validate before a full run

Since the loaders fail fast on the first malformed case (see [dataset-format.md](dataset-format.md)), it's worth a quick syntax/shape check before committing to a long batch run:

```bash
python -c "import json; json.load(open('data/datasets/NEW_DATASET_FOLDER/MY_NEW_SET.json'))"
```

Then do a small trial run with `--limit` before running the full dataset and every method:

```bash
python scripts/evaluation/run_set_based.py \
    --test-set data/datasets/NEW_DATASET_FOLDER/MY_NEW_SET.json \
    --limit 5
```

Check the printed per-case lines and confirm `outputs/evaluation/MY_NEW_SET/cache/case_0000.json` (etc.) looks correct before running the rest of the methods and the full case count.


## 5. Run the full workflow

Once the trial run looks correct, follow the standard sequence in [workflow-overview.md](workflow-overview.md): run each applicable batch runner over the full dataset (no `--limit`), then run the evaluator:

```bash
python scripts/evaluation/run_set_based.py --test-set data/datasets/NEW_DATASET_FOLDER/MY_NEW_SET.json
python scripts/evaluation/run_tfidf.py --test-set data/datasets/NEW_DATASET_FOLDER/MY_NEW_SET.json
python scripts/evaluation/run_semantic.py --test-set data/datasets/NEW_DATASET_FOLDER/MY_NEW_SET.json
# ...remaining runners as needed

python scripts/evaluation/evaluator.py --dataset MY_NEW_SET
```


## 6. Document the dataset

Once a new dataset is in regular use, add it to [dataset-available.md](dataset-available.md) with a brief description and its source, so other contributors know what it covers and don't duplicate it.
