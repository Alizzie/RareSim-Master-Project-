# PhenoBrain (Local)

A self-hosted deployment of PhenoBrain, built from the authors' GitHub pipeline
and run against our own datasets on the `INTEGRATE_CCRD_OMIM_ORPHA` knowledge
base. Unlike the [hosted PhenoBrain](./phenobrain) API, the local pipeline runs
the models directly and writes a **raw per-model rank CSV**. Our
`run_phenobrain_local.py` script then standardizes that CSV into the same
summary TSV format used by every other tool.

- **Repository:** [xiaohaomao/timgroup_disease_diagnosis](https://github.com/xiaohaomao/timgroup_disease_diagnosis)
- **Evaluation entry point:** `core/script/test/test_optimal_model.py`
- **Knowledge base:** `INTEGRATE_CCRD_OMIM_ORPHA` (pinned 2019 HPO release)
- **Tested on:** Linux, May 2026

> **Important:** The pipeline is pinned to a **2019 HPO ontology release** and
> uses internal `RD:` disease codes. Custom datasets that contain newer HPO
> terms or diseases missing from the pinned knowledge base get silently filtered
> out (see Troubleshooting). This is why our `0.1.27` dataset dropped from
> **10,374 → 6,971 cases** after correct filtering was applied.

---

## Requirements

- Python environment matching the repo's pinned dependencies (notably
  `scikit-learn==0.22.1` — see Troubleshooting #4)
- The `INTEGRATE_CCRD_OMIM_ORPHA` knowledge base and pretrained model artifacts
- `requests` (for the standardizer's `RD:` → OMIM/ORPHA remap)
- Internet access to `www.phenobrain.cs.tsinghua.edu.cn` for the remap step

---

## 1. Run PhenoBrain Locally

Follow the upstream repository's guidance to run the evaluation pipeline
(`test_optimal_model.py`) against your dataset on the `INTEGRATE_CCRD_OMIM_ORPHA`
knowledge base. This produces the raw results CSV — one row per patient case,
one column per model, where each cell is the **rank the model assigned to that
case's ground-truth disease**.

> This step is the prerequisite. Our code does **not** run PhenoBrain — it only
> standardizes the raw CSV it produces. See Troubleshooting below for the issues
> we hit getting this pipeline to run against custom datasets.

### Raw CSV shape

| Column group | Columns |
|--------------|---------|
| Fixed | `DATA_RANK`, `DISEASE_CODE`, `DISEASE_NAME`, `HPO_CODE`, `HPO_NAME` |
| Per-model rank | `MICA-QD-Random`, `BOQAModel-dp1.0-Random`, `RDDModel-Ances-Random`, `GDDPFisherModel-MinIC-Random`, `RBPModel-Random`, `MinIC-QD-Random`, `MICALin-QD-Random`, `MICAJC-QD-Random`, `SimGICModel-Random`, `JaccardModel-Random`, `SimTOModel-Random`, `CosineModel-Random`, `ICTODQAcross-Ave-Random`, `HPOProbMNB-Random`, `CNB-Random`, `NN-Mixup-Random-1` |

`DISEASE_CODE` and `HPO_CODE` are stringified Python lists (e.g. `['RD:6786']`).
Columns are tab-separated because the list cells contain unquoted commas.

---

## 2. Standardize the Raw CSV

<!-- % default behavior changed: standardizes every model column in one pass instead of requiring one --model per run -->
```bash
# Standardize EVERY model column in one pass (default — no model flag needed)
python3 run_phenobrain_local.py --input raw/0.1.27_raw.csv --dataset 0.1.27

# List the model columns present in a raw CSV first, if you just want to check
python3 run_phenobrain_local.py --input raw/0.1.27_raw.csv --list-models

# Restrict to one or more specific model columns
python3 run_phenobrain_local.py \
  --input raw/0.1.27_raw.csv \
  --dataset 0.1.27 \
  --models NN-Mixup-Random-1 MICA-QD-Random

# Skip the RD -> OMIM/ORPHA API remap (offline; keep raw RD codes)
python3 run_phenobrain_local.py \
  --input raw/0.1.27_raw.csv \
  --dataset 0.1.27 \
  --no-remap
```

Each model is written to its own `phenobrain (<model>)_benchmarks/` folder, so
`compare_methods.py`'s auto-discovery (folder name minus `_benchmarks` =
method name) picks up every model as a separate method automatically — e.g.
`phenobrain (NN-Mixup-Random-1)`, `phenobrain (MICA-QD-Random)`, etc. There's
no need to name models individually unless you want to restrict the run.

### Arguments

<!-- % --model replaced with --models (defaults to all); --out-dir is now a base dir -->
| Argument | Description |
|----------|-------------|
| `--input` | Path to the raw PhenoBrain CSV/TSV. Required. |
| `--dataset` | Dataset name (used for `case_id` prefix and output filename). Required unless `--list-models`. |
| `--models` | One or more model rank columns to standardize (see `--list-models`). Default: every model column found in `--input`. |
| `--out-dir` | Base output directory; each model gets its own `phenobrain (<model>)_benchmarks` subfolder here. Default: `output/validation_tools`. |
| `--delimiter` | Column delimiter. Auto-detected (tab vs comma) when omitted. |
| `--topk` | Treat a ground-truth rank greater than this as not found. Default: keep all ranks. |
| `--no-remap` | Skip the `RD:` → OMIM/ORPHA API remap and keep raw RD codes (offline mode). |
| `--list-models` | Print the model columns found in `--input` and exit. |

---

## Implementation

`run_phenobrain_local.py` is a pure standardizer — no PhenoBrain execution:

1. **Read.** Loads the raw CSV (auto-detecting tab vs comma) and parses the
   stringified-list cells `DISEASE_CODE` and `HPO_CODE`.
2. **Remap.** Collects every ground-truth `RD:` code and maps it to OMIM/ORPHA
   in batches via the same `disease-list-detail` API used by the hosted runner
   (`run_phenobrain.create_RD_code_mapper`). `--no-remap` skips this and keeps
   the raw `RD:` codes. This runs **once** per invocation, and the resulting
   mapping is reused across every model — not recomputed per model. <!-- % clarify mapper is shared across models -->
3. **Standardize per model.** Every model column (or the subset passed to
   `--models`) is standardized in turn: `rank` is read directly from that
   model's column (the cell already *is* the ground-truth rank). `case_id` is
   `<dataset>_case_<DATA_RANK:04d>`, matching the other runners' convention so
   `compare_methods.py` can align cases. <!-- % now loops over all models instead of one --model per run -->
4. **Write.** Each model's summary is written to its own
   `phenobrain (<model>)_benchmarks/<dataset>_summary.tsv`, plus a matching
   `<dataset>_stats.txt` — so `compare_methods.py` treats each model as an
   independent method with no extra registration step. <!-- % per-model output folders -->

> **Not available from the raw CSV:** `score` and `query_time_sec` are recorded
> as `None` — the raw cells are ranks, not scores, and the local pipeline does
> not emit per-case timings.

> **Alignment caveat:** because filtering dropped ~3.4k cases from `0.1.27`, the
> `DATA_RANK` indices may not line up 1:1 with the same dataset run through the
> other tools. Treat cross-tool alignment for this dataset with care.

---

## Output Format

<!-- % results now written per-model, one folder per model, instead of a single shared folder -->
Each model's results are written to its own
`output/validation_tools/phenobrain (<model>)_benchmarks/<dataset>_summary.tsv`,
e.g. `output/validation_tools/phenobrain (NN-Mixup-Random-1)_benchmarks/0.1.27_summary.tsv`:

| Column | Description |
|--------|-------------|
| `case_id` | Patient case identifier (`<dataset>_case_NNNN`) |
| `n_hpo` | Number of HPO terms in the case |
| `confirmed_diseases` | Ground-truth disease ID(s), remapped to OMIM/ORPHA (or raw `RD:` with `--no-remap`) |
| `rank` | Ground-truth rank from the selected model column (`None` if beyond `--topk`) |
| `matched_id` | Same as `confirmed_diseases` when found, else `None` |
| `score` | `None` (not available in the raw CSV) |
| `status` | `True` for every parsed row |
| `query_time_sec` | `None` (not available in the raw CSV) |

---

## Results

<!-- % note added: results are now generated per model; table below is a template to duplicate per model, or extend with a Model column, once real numbers exist -->
> Since every model column now produces its own summary, this table is a
> template — duplicate it per model (or add a `Model` column) once real
> numbers are filled in.

Benchmark results (metrics pending — placeholders shown as `/`):

| Dataset | n | Found | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 | Avg. query time (s) |
|---------|---|-------|-------|-------|-------|--------|--------|---------------------|
| 0.1.27 | 6971 | / | / | / | / | / | / | / |
| MME | / | / | / | / | / | / | / | / |
| HMS | / | / | / | / | / | / | / | / |
| LIRICAL | / | / | / | / | / | / | / | / |
| RAMEDIS | / | / | / | / | / | / | / | / |
| PUMCH_L | / | / | / | / | / | / | / | / |
| PUMCH-ADM | / | / | / | / | / | / | / | / |

> `0.1.27` was reduced from 10,374 to 6,971 cases after correct HPO/disease
> filtering (see Troubleshooting). Query-time figures are not produced by the
> local pipeline and are always `/` for this tool.

---

## Troubleshooting

Compact index of issues hit getting `test_optimal_model.py` to run against
custom datasets on `INTEGRATE_CCRD_OMIM_ORPHA`. The shared root cause: **custom
datasets contain disease or HPO codes absent from the pinned 2019 knowledge
base, and the harness load path (`ModelTestor.load_test_data`) always calls
`get_dataset(..., filter=False)`, so existing filters never run.**

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `Loaded Empty Dataset` / `SIZE=0` | Dataset name isn't a key in `DataHelper.test_to_path` (it's a fixed dict, not a folder scan) | Register the name in `test_names` + `test_to_path` |
| 2 | File present but not found | Resolved path is `data/preprocess/patient/<mark>/test/<name>.json` (with `INTEGRATE_` stripped) | Move/copy the JSON to the exact resolved path |
| 3 | `AttributeError: ... 'dis_filter'` | `dis_filter` is referenced but never defined | Add `dis_filter` to `DataHelper` (drop disease codes not in the KB) |
| 4 | `CNB.joblib` load warning/failure | Model pickled with scikit-learn 0.22.1; env pins 0.21.3 | `pip install scikit-learn==0.22.1` (watch other pickled models) |
| 5 | `IndexError: Cannot choose from an empty sequence` (`get_ytrue`) | Patients left with empty disease list after code conversion | Resolved by #3's `dis_filter`; also guard `get_ytrue` for empty input |
| 6 | Same `IndexError`, large dataset | Harness loads with `filter=False`; `remove_dis_map_general` can empty a disease list post-load | Drop empty-disease patients where `self.data[name]` is finalized (not just in `DataHelper`) |
| 7 | `ValueError: not enough values to unpack (expected 2, got 0)` (`PUMCH-L`) | Bare name isn't a registered key; `zip(*[])` fails | Register the exact name (e.g. `PUMCH_L`) or use a registered variant |
| 8 | Run appears stuck ~24 h after finishing | `__del__` teardown hang (TF 1.x / multiprocessing pool cleanup); results already saved | Confirm via saved files, kill the process, end scripts with `os._exit(0)` |
| 9 | `KeyError: 'HP:...'` during BOQA scoring | HPO code not in the pinned 2019 ontology; graph walk fails instead of skipping | Add an `hpo_filter` (old→new HPO map) and wire it into the actual load path |

**Recommended sanity check for any new dataset** (catches the above in seconds
instead of after a multi-hour run):

```python
raw  = dh.get_dataset(name, 'test', filter=False)
kept = dh.get_dataset(name, 'test', filter=True)   # after the fixes above
print('raw:', len(raw), 'kept:', len(kept))
# also check unknown HPO codes against get_slice_hpo_dict()
```

For `0.1.27` this check reported the drop from 10,374 raw to 6,971 kept cases —
the filtered set is what feeds the standardizer.

---

## Reference

> Mao X. et al. *A phenotype-based AI pipeline outperforms human experts in differentially diagnosing rare diseases using EHRs.* npj Digital Medicine 8, 68 (2025). https://doi.org/10.1038/s41746-025-01452-1
