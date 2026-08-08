# PhenoBrain (Local)

A self-hosted deployment of PhenoBrain, built from the authors' GitHub pipeline
and run against our own datasets on the `INTEGRATE_CCRD_OMIM_ORPHA` knowledge
base. Unlike the [hosted PhenoBrain](./phenobrain) API, the local pipeline runs
the models directly and writes a **raw per-model rank export (`.xlsx`)**. Our <!-- % csv -> xlsx -->
`run_phenobrain_local.py` script then standardizes that export into the same
summary TSV format used by every other tool. <!-- % csv -> export -->

- **Repository:** [xiaohaomao/timgroup_disease_diagnosis](https://github.com/xiaohaomao/timgroup_disease_diagnosis)
- **Evaluation entry point:** `core/script/test/test_optimal_model.py`
- **Knowledge base:** `INTEGRATE_CCRD_OMIM_ORPHA` (pinned 2019 HPO release)
- **Tested on:** Linux, May 2026

> **Important:** The pipeline is pinned to a **2019 HPO ontology release** and
> uses internal `RD:` disease codes. Custom datasets that contain newer HPO
> terms or diseases missing from the pinned knowledge base get silently filtered
> out (see Troubleshooting). This is why our `0.1.27` dataset dropped from
> **10,374 → 6,971 cases** after correct filtering was applied.

## 1. Requirements

- Python environment matching the repo's pinned dependencies (notably
  `scikit-learn==0.22.1` — see Troubleshooting #4)
- The `INTEGRATE_CCRD_OMIM_ORPHA` knowledge base and pretrained model artifacts
- `requests` (for the standardizer's `RD:` → OMIM/ORPHA remap)
- Internet access to `www.phenobrain.cs.tsinghua.edu.cn` for the remap step

## 2. Run PhenoBrain Locally

Follow the upstream repository's guidance to run the evaluation pipeline
(`test_optimal_model.py`) against your dataset on the `INTEGRATE_CCRD_OMIM_ORPHA`
knowledge base. This produces the raw results export (`.xlsx`) — one row per
patient case, one column per model, where each cell is the **rank the model
assigned to that case's ground-truth disease**.

> This step is the prerequisite. Our code does **not** run PhenoBrain. It only
> standardizes the raw export it produces. See Troubleshooting below for the 
> issues we hit getting this pipeline to run against custom datasets.

### Raw export shape 

| Column group | Columns |
|--------------|---------|
| Fixed | `DATA_RANK`, `DISEASE_CODE`, `DISEASE_NAME`, `HPO_CODE`, `HPO_NAME` |
| Per-model rank | `MICA-QD-Random`, `BOQAModel-dp1.0-Random`, `RDDModel-Ances-Random`, `GDDPFisherModel-MinIC-Random`, `RBPModel-Random`, `MinIC-QD-Random`, `MICALin-QD-Random`, `MICAJC-QD-Random`, `SimGICModel-Random`, `JaccardModel-Random`, `SimTOModel-Random`, `CosineModel-Random`, `ICTODQAcross-Ave-Random`, `HPOProbMNB-Random`, `CNB-Random`, `NN-Mixup-Random-1` |

`DISEASE_CODE` and `HPO_CODE` are stringified Python lists (e.g. `['RD:6786']`). 
The export is a native `.xlsx` workbook — the standardizer reads the first
sheet directly (via `openpyxl`) with no delimiter to worry about. A CSV/TSV
export also still works if that's what your run produced (delimiter is
auto-detected).


## 3. Standardize the Raw Export 

```bash
# Standardize EVERY model column in one pass (default — no model flag needed)
python3 run_phenobrain_local.py --input raw/0.1.27_raw.xlsx --dataset 0.1.27

# List the model columns present in a raw export first, if you just want to check
python3 run_phenobrain_local.py --input raw/0.1.27_raw.xlsx --list-models

# Restrict to one or more specific model columns
python3 run_phenobrain_local.py \
  --input raw/0.1.27_raw.xlsx \
  --dataset 0.1.27 \
  --models NN-Mixup-Random-1 MICA-QD-Random

# Skip the RD -> OMIM/ORPHA API remap (offline; keep raw RD codes)
python3 run_phenobrain_local.py \
  --input raw/0.1.27_raw.xlsx \
  --dataset 0.1.27 \
  --no-remap

# A raw CSV/TSV export also still works
python3 run_phenobrain_local.py --input raw/0.1.27_raw.csv --dataset 0.1.27
```

Each model is written to its own `phenobrain (<model>)_benchmarks/` folder, so
`compare_methods.py`'s auto-discovery (folder name minus `_benchmarks` =
method name) picks up every model as a separate method automatically — e.g.
`phenobrain (NN-Mixup-Random-1)`, `phenobrain (MICA-QD-Random)`, etc. There's
no need to name models individually unless you want to restrict the run.

### Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to the raw PhenoBrain export: `.xlsx` (default local export format), or `.csv`/`.tsv`. Required. | <!-- % csv/tsv -> xlsx as default, csv/tsv still supported -->
| `--dataset` | Dataset name (used for `case_id` prefix and output filename). Required unless `--list-models`. |
| `--models` | One or more model rank columns to standardize (see `--list-models`). Default: every model column found in `--input`. |
| `--out-dir` | Base output directory; each model gets its own `phenobrain (<model>)_benchmarks` subfolder here. Default: `output/validation_tools`. |
| `--delimiter` | Column delimiter for CSV/TSV input (ignored for `.xlsx`). Auto-detected (tab vs comma) when omitted. | <!-- % clarified: ignored for xlsx -->
| `--topk` | Treat a ground-truth rank greater than this as not found. Default: keep all ranks. |
| `--no-remap` | Skip the `RD:` → OMIM/ORPHA API remap and keep raw RD codes (offline mode). |
| `--list-models` | Print the model columns found in `--input` and exit. |


## 4. Implementation

`run_phenobrain_local.py` is a pure standardizer — no PhenoBrain execution:

1. **Read.** Loads the raw export — `.xlsx` via `openpyxl` (first sheet), or 
   CSV/TSV (auto-detecting tab vs comma) as a fallback — and parses the
   stringified-list cells `DISEASE_CODE` and `HPO_CODE`.
2. **Remap.** Collects every ground-truth `RD:` code and maps it to OMIM/ORPHA
   in batches via the same `disease-list-detail` API used by the hosted runner
   (`run_phenobrain.create_RD_code_mapper`). `--no-remap` skips this and keeps
   the raw `RD:` codes. This runs **once** per invocation, and the resulting
   mapping is reused across every model — not recomputed per model. 
3. **Standardize per model.** Every model column (or the subset passed to
   `--models`) is standardized in turn: `rank` is read directly from that
   model's column (the cell already *is* the ground-truth rank). `case_id` is
   `<dataset>_case_<DATA_RANK:04d>`, matching the other runners' convention so
   `compare_methods.py` can align cases. 
4. **Write.** Each model's summary is written to its own
   `phenobrain (<model>)_benchmarks/<dataset>_summary.tsv`, plus a matching
   `<dataset>_stats.txt` — so `compare_methods.py` treats each model as an
   independent method with no extra registration step. 

> **Not available from the raw export:** `score` and `query_time_sec` are
> recorded as `None` — the raw cells are ranks, not scores, and the local
> pipeline does not emit per-case timings.

> **Alignment caveat:** because filtering dropped ~3.4k cases from `0.1.27`, the
> `DATA_RANK` indices may not line up 1:1 with the same dataset run through the
> other tools. Treat cross-tool alignment for this dataset with care.


## 5. Output Format

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
| `score` | `None` (not available in the raw export) |
| `status` | `True` for every parsed row |
| `query_time_sec` | `None` (not available in the raw export) |


## 6. Results

Benchmark results for every local model, per dataset (compare_methods.py output, May–Jul 2026). Query-time is always `/` — the local pipeline doesn't emit per-case timings.

### 0.1.27

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 6971 | 6971/6971 | 0.2246 | 0.4015 | 0.4691 | 0.3098 | / |
| CNB-Random | 6971 | 6971/6971 | 0.2631 | 0.4512 | 0.5342 | 0.3541 | / |
| CosineModel-Random | 6971 | 6971/6971 | 0.0782 | 0.1634 | 0.2330 | 0.1289 | / |
| GDDPFisherModel-MinIC-Random | 6971 | 6971/6971 | 0.1766 | 0.3757 | 0.4560 | 0.2689 | / |
| HPOProbMNB-Random | 6971 | 6971/6971 | 0.2586 | 0.4545 | 0.5292 | 0.3553 | / |
| ICTODQAcross-Ave-Random | 6971 | 6971/6971 | 0.2688 | 0.4570 | 0.5305 | 0.3598 | / |
| JaccardModel-Random | 6971 | 6971/6971 | 0.0657 | 0.1298 | 0.1795 | 0.1034 | / |
| MICA-QD-Random | 6971 | 6971/6971 | 0.2738 | 0.4453 | 0.5371 | 0.3612 | / |
| MICAJC-QD-Random | 6971 | 6971/6971 | 0.2671 | 0.4474 | 0.5298 | 0.3555 | / |
| MICALin-QD-Random | 6971 | 6971/6971 | 0.2505 | 0.4278 | 0.5146 | 0.3395 | / |
| MinIC-QD-Random | 6971 | 6971/6971 | 0.2767 | 0.4458 | 0.5302 | 0.3618 | / |
| NN-Mixup-Random-1 | 6971 | 6971/6971 | 0.2289 | 0.4309 | 0.5090 | 0.3263 | / |
| RBPModel-Random | 6971 | 6971/6971 | 0.2595 | 0.4169 | 0.4926 | 0.3386 | / |
| RDDModel-Ances-Random | 6971 | 6971/6971 | 0.0971 | 0.2153 | 0.3130 | 0.1663 | / |
| SimGICModel-Random | 6971 | 6971/6971 | 0.0838 | 0.1601 | 0.2223 | 0.1307 | / |
| SimTOModel-Random | 6971 | 6971/6971 | 0.2541 | 0.4209 | 0.5032 | 0.3381 | / |

### HMS

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 88 | 88/88 | 0.1250 | 0.3182 | 0.4659 | 0.2203 | / |
| CNB-Random | 88 | 88/88 | 0.1136 | 0.2955 | 0.4205 | 0.2140 | / |
| CosineModel-Random | 88 | 88/88 | 0.0795 | 0.1364 | 0.1932 | 0.1277 | / |
| GDDPFisherModel-MinIC-Random | 88 | 88/88 | 0.0682 | 0.2500 | 0.4205 | 0.1671 | / |
| HPOProbMNB-Random | 88 | 88/88 | 0.1932 | 0.4432 | 0.5455 | 0.3122 | / |
| ICTODQAcross-Ave-Random | 88 | 88/88 | 0.1477 | 0.3977 | 0.5455 | 0.2742 | / |
| JaccardModel-Random | 88 | 88/88 | 0.0682 | 0.1364 | 0.1705 | 0.1134 | / |
| MICA-QD-Random | 88 | 88/88 | 0.1705 | 0.2955 | 0.4318 | 0.2503 | / |
| MICAJC-QD-Random | 88 | 88/88 | 0.1932 | 0.3523 | 0.4886 | 0.2747 | / |
| MICALin-QD-Random | 88 | 88/88 | 0.1591 | 0.2955 | 0.4205 | 0.2361 | / |
| MinIC-QD-Random | 88 | 88/88 | 0.1477 | 0.3636 | 0.5000 | 0.2600 | / |
| NN-Mixup-Random-1 | 88 | 88/88 | 0.1477 | 0.2955 | 0.3977 | 0.2284 | / |
| RBPModel-Random | 88 | 88/88 | 0.1250 | 0.3295 | 0.4432 | 0.2315 | / |
| RDDModel-Ances-Random | 88 | 88/88 | 0.0568 | 0.1250 | 0.1818 | 0.1100 | / |
| SimGICModel-Random | 88 | 88/88 | 0.0795 | 0.1818 | 0.2159 | 0.1359 | / |
| SimTOModel-Random | 88 | 88/88 | 0.1023 | 0.2045 | 0.3068 | 0.1681 | / |

### LIRICAL

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 370 | 370/370 | 0.3270 | 0.5027 | 0.5541 | 0.4096 | / |
| CNB-Random | 370 | 370/370 | 0.3243 | 0.5081 | 0.5973 | 0.4133 | / |
| CosineModel-Random | 370 | 370/370 | 0.1838 | 0.3135 | 0.3757 | 0.2502 | / |
| GDDPFisherModel-MinIC-Random | 370 | 370/370 | 0.2568 | 0.4243 | 0.5081 | 0.3388 | / |
| HPOProbMNB-Random | 370 | 370/370 | 0.3405 | 0.5189 | 0.5919 | 0.4227 | / |
| ICTODQAcross-Ave-Random | 370 | 370/370 | 0.3405 | 0.5243 | 0.6162 | 0.4288 | / |
| JaccardModel-Random | 370 | 370/370 | 0.1459 | 0.2676 | 0.3216 | 0.2052 | / |
| MICA-QD-Random | 370 | 370/370 | 0.3216 | 0.4784 | 0.5595 | 0.4018 | / |
| MICAJC-QD-Random | 370 | 370/370 | 0.3135 | 0.4838 | 0.5595 | 0.3963 | / |
| MICALin-QD-Random | 370 | 370/370 | 0.2649 | 0.4378 | 0.5189 | 0.3527 | / |
| MinIC-QD-Random | 370 | 370/370 | 0.3135 | 0.5135 | 0.6027 | 0.4117 | / |
| NN-Mixup-Random-1 | 370 | 370/370 | 0.3000 | 0.4784 | 0.6027 | 0.3907 | / |
| RBPModel-Random | 370 | 370/370 | 0.2919 | 0.4649 | 0.5459 | 0.3766 | / |
| RDDModel-Ances-Random | 370 | 370/370 | 0.1919 | 0.3432 | 0.4351 | 0.2724 | / |
| SimGICModel-Random | 370 | 370/370 | 0.1973 | 0.3270 | 0.3757 | 0.2595 | / |
| SimTOModel-Random | 370 | 370/370 | 0.2649 | 0.4270 | 0.5081 | 0.3460 | / |

### MME

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 40 | 40/40 | 0.4500 | 0.6250 | 0.6500 | 0.5422 | / |
| CNB-Random | 40 | 40/40 | 0.5000 | 0.6500 | 0.7000 | 0.5787 | / |
| CosineModel-Random | 40 | 40/40 | 0.2250 | 0.3500 | 0.4250 | 0.2870 | / |
| GDDPFisherModel-MinIC-Random | 40 | 40/40 | 0.3750 | 0.5000 | 0.6000 | 0.4355 | / |
| HPOProbMNB-Random | 40 | 40/40 | 0.6250 | 0.8000 | 0.8000 | 0.6963 | / |
| ICTODQAcross-Ave-Random | 40 | 40/40 | 0.5500 | 0.8000 | 0.8500 | 0.6712 | / |
| JaccardModel-Random | 40 | 40/40 | 0.2000 | 0.3250 | 0.3250 | 0.2467 | / |
| MICA-QD-Random | 40 | 40/40 | 0.3000 | 0.6500 | 0.7250 | 0.4662 | / |
| MICAJC-QD-Random | 40 | 40/40 | 0.4750 | 0.7000 | 0.8250 | 0.5958 | / |
| MICALin-QD-Random | 40 | 40/40 | 0.4250 | 0.6750 | 0.7000 | 0.5284 | / |
| MinIC-QD-Random | 40 | 40/40 | 0.4000 | 0.7000 | 0.7250 | 0.5418 | / |
| NN-Mixup-Random-1 | 40 | 40/40 | 0.4000 | 0.7000 | 0.7250 | 0.5205 | / |
| RBPModel-Random | 40 | 40/40 | 0.2500 | 0.6000 | 0.6750 | 0.3968 | / |
| RDDModel-Ances-Random | 40 | 40/40 | 0.3250 | 0.4750 | 0.5250 | 0.4017 | / |
| SimGICModel-Random | 40 | 40/40 | 0.2000 | 0.3250 | 0.3750 | 0.2526 | / |
| SimTOModel-Random | 40 | 40/40 | 0.4250 | 0.5500 | 0.6500 | 0.4968 | / |

### RAMEDIS

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 375 | 375/375 | 0.1387 | 0.2747 | 0.3147 | 0.2003 | / |
| CNB-Random | 375 | 375/375 | 0.2053 | 0.5360 | 0.6613 | 0.3591 | / |
| CosineModel-Random | 375 | 375/375 | 0.0373 | 0.1040 | 0.1413 | 0.0758 | / |
| GDDPFisherModel-MinIC-Random | 375 | 375/375 | 0.1173 | 0.3547 | 0.4987 | 0.2345 | / |
| HPOProbMNB-Random | 375 | 375/375 | 0.2400 | 0.5413 | 0.6293 | 0.3730 | / |
| ICTODQAcross-Ave-Random | 375 | 375/375 | 0.2107 | 0.4987 | 0.6293 | 0.3385 | / |
| JaccardModel-Random | 375 | 375/375 | 0.0320 | 0.0827 | 0.1040 | 0.0600 | / |
| MICA-QD-Random | 375 | 375/375 | 0.1227 | 0.2773 | 0.3973 | 0.2124 | / |
| MICAJC-QD-Random | 375 | 375/375 | 0.1787 | 0.3627 | 0.4480 | 0.2716 | / |
| MICALin-QD-Random | 375 | 375/375 | 0.0960 | 0.2187 | 0.3227 | 0.1669 | / |
| MinIC-QD-Random | 375 | 375/375 | 0.2053 | 0.4827 | 0.5920 | 0.3282 | / |
| NN-Mixup-Random-1 | 375 | 375/375 | 0.2080 | 0.4827 | 0.6453 | 0.3329 | / |
| RBPModel-Random | 375 | 375/375 | 0.2187 | 0.5547 | 0.6613 | 0.3689 | / |
| RDDModel-Ances-Random | 375 | 375/375 | 0.0347 | 0.1147 | 0.1973 | 0.0905 | / |
| SimGICModel-Random | 375 | 375/375 | 0.0533 | 0.1600 | 0.2240 | 0.1108 | / |
| SimTOModel-Random | 375 | 375/375 | 0.1280 | 0.3867 | 0.5440 | 0.2547 | / |

### PUMCH_L

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 988 | 988/988 | 0.2652 | 0.4464 | 0.5395 | 0.3540 | / |
| CNB-Random | 988 | 988/988 | 0.2925 | 0.5121 | 0.6012 | 0.3918 | / |
| CosineModel-Random | 988 | 988/988 | 0.2702 | 0.4626 | 0.5455 | 0.3663 | / |
| GDDPFisherModel-MinIC-Random | 988 | 988/988 | 0.2115 | 0.4160 | 0.5304 | 0.3113 | / |
| HPOProbMNB-Random | 988 | 988/988 | 0.2460 | 0.4534 | 0.5648 | 0.3486 | / |
| ICTODQAcross-Ave-Random | 988 | 988/988 | 0.2864 | 0.5283 | 0.6144 | 0.3956 | / |
| JaccardModel-Random | 988 | 988/988 | 0.2874 | 0.4595 | 0.5466 | 0.3731 | / |
| MICA-QD-Random | 988 | 988/988 | 0.2257 | 0.4109 | 0.5314 | 0.3217 | / |
| MICAJC-QD-Random | 988 | 988/988 | 0.2399 | 0.4453 | 0.5344 | 0.3362 | / |
| MICALin-QD-Random | 988 | 988/988 | 0.1366 | 0.3391 | 0.4332 | 0.2382 | / |
| MinIC-QD-Random | 988 | 988/988 | 0.2611 | 0.4777 | 0.5779 | 0.3651 | / |
| NN-Mixup-Random-1 | 988 | 988/988 | 0.2955 | 0.5020 | 0.5941 | 0.3940 | / |
| RBPModel-Random | 988 | 988/988 | 0.1721 | 0.3644 | 0.4686 | 0.2713 | / |
| RDDModel-Ances-Random | 988 | 988/988 | 0.1549 | 0.2976 | 0.3704 | 0.2224 | / |
| SimGICModel-Random | 988 | 988/988 | 0.2986 | 0.4899 | 0.5860 | 0.3936 | / |
| SimTOModel-Random | 988 | 988/988 | 0.1275 | 0.3036 | 0.4130 | 0.2156 | / |

### PUMCH-ADM

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 75 | 75/75 | 0.2000 | 0.3867 | 0.4800 | 0.2953 | / |
| CNB-Random | 75 | 75/75 | 0.3200 | 0.5333 | 0.6667 | 0.4201 | / |
| CosineModel-Random | 75 | 75/75 | 0.0667 | 0.1733 | 0.2533 | 0.1390 | / |
| GDDPFisherModel-MinIC-Random | 75 | 75/75 | 0.2667 | 0.5200 | 0.6667 | 0.3825 | / |
| HPOProbMNB-Random | 75 | 75/75 | 0.4267 | 0.6400 | 0.6800 | 0.5200 | / |
| ICTODQAcross-Ave-Random | 75 | 75/75 | 0.3600 | 0.6000 | 0.6800 | 0.4759 | / |
| JaccardModel-Random | 75 | 75/75 | 0.0667 | 0.1467 | 0.2133 | 0.1186 | / |
| MICA-QD-Random | 75 | 75/75 | 0.3067 | 0.4800 | 0.5733 | 0.4033 | / |
| MICAJC-QD-Random | 75 | 75/75 | 0.3067 | 0.5600 | 0.6533 | 0.4198 | / |
| MICALin-QD-Random | 75 | 75/75 | 0.2667 | 0.5067 | 0.5600 | 0.3714 | / |
| MinIC-QD-Random | 75 | 75/75 | 0.2667 | 0.6000 | 0.6800 | 0.4046 | / |
| NN-Mixup-Random-1 | 75 | 75/75 | 0.3067 | 0.5600 | 0.6533 | 0.4231 | / |
| RBPModel-Random | 75 | 75/75 | 0.2533 | 0.5067 | 0.6267 | 0.3757 | / |
| RDDModel-Ances-Random | 75 | 75/75 | 0.0800 | 0.2667 | 0.3200 | 0.1681 | / |
| SimGICModel-Random | 75 | 75/75 | 0.0667 | 0.1733 | 0.2267 | 0.1357 | / |
| SimTOModel-Random | 75 | 75/75 | 0.2400 | 0.4533 | 0.5333 | 0.3440 | / |

### GA4GH_Phenopackets

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 384 | 384/384 | 0.3255 | 0.5078 | 0.5755 | 0.4118 | / |
| CNB-Random | 384 | 384/384 | 0.3281 | 0.5286 | 0.6380 | 0.4241 | / |
| CosineModel-Random | 384 | 384/384 | 0.1823 | 0.3125 | 0.3776 | 0.2495 | / |
| GDDPFisherModel-MinIC-Random | 384 | 384/384 | 0.2734 | 0.4505 | 0.5443 | 0.3598 | / |
| HPOProbMNB-Random | 384 | 384/384 | 0.3542 | 0.5208 | 0.6198 | 0.4355 | / |
| ICTODQAcross-Ave-Random | 384 | 384/384 | 0.3516 | 0.5443 | 0.6484 | 0.4439 | / |
| JaccardModel-Random | 384 | 384/384 | 0.1432 | 0.2734 | 0.3255 | 0.2052 | / |
| MICA-QD-Random | 384 | 384/384 | 0.3281 | 0.4974 | 0.6016 | 0.4157 | / |
| MICAJC-QD-Random | 384 | 384/384 | 0.3281 | 0.5104 | 0.5938 | 0.4143 | / |
| MICALin-QD-Random | 384 | 384/384 | 0.2760 | 0.4557 | 0.5599 | 0.3717 | / |
| MinIC-QD-Random | 384 | 384/384 | 0.3281 | 0.5391 | 0.6536 | 0.4297 | / |
| NN-Mixup-Random-1 | 384 | 384/384 | 0.3151 | 0.4948 | 0.6250 | 0.4071 | / |
| RBPModel-Random | 384 | 384/384 | 0.2995 | 0.4792 | 0.5599 | 0.3845 | / |
| RDDModel-Ances-Random | 384 | 384/384 | 0.1901 | 0.3411 | 0.4323 | 0.2711 | / |
| SimGICModel-Random | 384 | 384/384 | 0.1953 | 0.3281 | 0.3750 | 0.2587 | / |
| SimTOModel-Random | 384 | 384/384 | 0.2708 | 0.4427 | 0.5417 | 0.3575 | / |

### MyGene2

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 146 | 146/146 | 0.2397 | 0.4041 | 0.5342 | 0.3261 | / |
| CNB-Random | 146 | 146/146 | 0.3699 | 0.5685 | 0.6438 | 0.4783 | / |
| CosineModel-Random | 146 | 146/146 | 0.1164 | 0.1164 | 0.1370 | 0.1306 | / |
| GDDPFisherModel-MinIC-Random | 146 | 146/146 | 0.1849 | 0.3288 | 0.3699 | 0.2489 | / |
| HPOProbMNB-Random | 146 | 146/146 | 0.3630 | 0.4863 | 0.5753 | 0.4351 | / |
| ICTODQAcross-Ave-Random | 146 | 146/146 | 0.2740 | 0.5342 | 0.6438 | 0.3859 | / |
| JaccardModel-Random | 146 | 146/146 | 0.1096 | 0.1164 | 0.1233 | 0.1170 | / |
| MICA-QD-Random | 146 | 146/146 | 0.3630 | 0.5890 | 0.6575 | 0.4620 | / |
| MICAJC-QD-Random | 146 | 146/146 | 0.2945 | 0.4863 | 0.5822 | 0.3974 | / |
| MICALin-QD-Random | 146 | 146/146 | 0.2603 | 0.5479 | 0.6027 | 0.3744 | / |
| MinIC-QD-Random | 146 | 146/146 | 0.3699 | 0.6096 | 0.6918 | 0.4739 | / |
| NN-Mixup-Random-1 | 146 | 146/146 | 0.2055 | 0.4658 | 0.5616 | 0.3041 | / |
| RBPModel-Random | 146 | 146/146 | 0.2671 | 0.4452 | 0.4863 | 0.3554 | / |
| RDDModel-Ances-Random | 146 | 146/146 | 0.1164 | 0.1507 | 0.1986 | 0.1529 | / |
| SimGICModel-Random | 146 | 146/146 | 0.1164 | 0.1164 | 0.1370 | 0.1269 | / |
| SimTOModel-Random | 146 | 146/146 | 0.3630 | 0.5548 | 0.6370 | 0.4601 | / |

### Test_Medical_Cases

| Model | n | Found | R@1 | R@5 | R@10 | MRR | Avg. query time (s) |
|-------|---|-------|-----|-----|------|-----|----------------------|
| BOQAModel-dp1.0-Random | 200 | 200/200 | 0.8550 | 0.9250 | 0.9500 | 0.8886 | / |
| CNB-Random | 200 | 200/200 | 0.6850 | 0.7950 | 0.8450 | 0.7403 | / |
| CosineModel-Random | 200 | 200/200 | 0.8600 | 0.9150 | 0.9300 | 0.8868 | / |
| GDDPFisherModel-MinIC-Random | 200 | 200/200 | 0.6600 | 0.8350 | 0.8600 | 0.7323 | / |
| HPOProbMNB-Random | 200 | 200/200 | 0.7800 | 0.8900 | 0.9250 | 0.8278 | / |
| ICTODQAcross-Ave-Random | 200 | 200/200 | 0.8000 | 0.9050 | 0.9250 | 0.8501 | / |
| JaccardModel-Random | 200 | 200/200 | 0.8200 | 0.8700 | 0.8900 | 0.8438 | / |
| MICA-QD-Random | 200 | 200/200 | 0.5250 | 0.6800 | 0.7300 | 0.5989 | / |
| MICAJC-QD-Random | 200 | 200/200 | 0.7850 | 0.8650 | 0.9000 | 0.8226 | / |
| MICALin-QD-Random | 200 | 200/200 | 0.5150 | 0.6500 | 0.7100 | 0.5803 | / |
| MinIC-QD-Random | 200 | 200/200 | 0.7450 | 0.8700 | 0.9000 | 0.8052 | / |
| NN-Mixup-Random-1 | 200 | 200/200 | 0.6650 | 0.8100 | 0.8300 | 0.7308 | / |
| RBPModel-Random | 200 | 200/200 | 0.7450 | 0.8150 | 0.8500 | 0.7805 | / |
| RDDModel-Ances-Random | 200 | 200/200 | 0.7950 | 0.8600 | 0.8700 | 0.8255 | / |
| SimGICModel-Random | 200 | 200/200 | 0.8650 | 0.9300 | 0.9500 | 0.8983 | / |
| SimTOModel-Random | 200 | 200/200 | 0.3900 | 0.5450 | 0.6300 | 0.4663 | / |

> `0.1.27` was reduced from 10,374 to 6,971 cases after correct HPO/disease
> filtering (see Troubleshooting). Query-time figures are not produced by the
> local pipeline and are always `/` for every model.


## 7. Troubleshooting

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

## 8. Reference

> Mao X. et al. *A phenotype-based AI pipeline outperforms human experts in differentially diagnosing rare diseases using EHRs.* npj Digital Medicine 8, 68 (2025). https://doi.org/10.1038/s41746-025-01452-1
