# Benchmark Dataset Standardization

## Purpose

Because the five benchmark sources do not arrive in one common format, four of them are converted by dedicated standardization scripts. Regardless of which script produced it, the output converges on the same shape: a JSON list of two-element entries per case,

```json
[
  [["<HPO ID>", "..."], ["<disease ID>", "..."]],
  [["<HPO ID>", "..."], ["<disease ID>", "..."]]
]
```

## PhenoBrain Benchmark

No standardization script exists for this source, because none is needed. The six PhenoBrain cohorts arrive already as HPO term lists paired directly with ground-truth disease identifiers, so no format conversion is required before reaching the evaluation pipeline.

## Medical Cases

Source file: a flat mapping from bare ORPHA code to raw clinical text.

```json
{
  "ORPHA_CODE": "clinical text",
  "...": "..."
}
```

Standardization script:

```text
run_medical_cases_extraction.py
```

For each entry, this script calls:

```python
build_patient_profile()
```

(see [Patient Profile Construction](patient-profile-construction.md)) using one or more of the extraction strategies listed there, selectable per run via:

```bash
--methods
```

Ground truth is constructed directly from the source key as a single-element list:

```text
["ORPHA:{code}"]
```

This is why Medical cases has exactly one ground-truth disease per case and only the ORPHA namespace throughout.

## Phenopacket Store & GA4GH Phenopackets

Both datasets are produced from the same raw phenopacket collections, using two closely related scripts.

### `standardize_phenopackets.py` (positive-only)

Creates the positive-only evaluation files used by the standard similarity methods.

- Reads HPO terms from each phenopacket's `phenotypicFeatures` list.
- Takes `type.id` from every feature **except** those flagged `excluded: true`.
- Excluded findings are omitted entirely from the positive-only output.

### `standardize_phenopackets_with_excluded.py` (negative-aware)

Processes the same source files but preserves the observed/excluded distinction.

- Direct observed terms → `hpo_terms`
- Explicitly absent findings → `excluded_hpo_terms`
- Reference diagnoses → `disease_codes`

This negative-aware output is used only for evaluating `set_jaccard_penalized`. The original positive-only files remain unchanged for all other methods.

## MyGene2

Source: JSON Lines (one JSON object per line), read from `mygene2_5.7.22.txt` — not a phenopacket format.

Standardization script:

```text
standardize_mygene2.py
```

- HPO terms come from the `positive_phenotypes` field, filtered to entries that literally start with `"HP:"`.
- Ground truth combines:
  - `orpha_id` — may be absent, a single value, or a list; all three forms are handled.
  - `omim` — a single value.
  - These are combined into separately prefixed `ORPHA:` and `OMIM:` codes, deduplicated and sorted together.
- A case is dropped if either the HPO term list or the disease code list is empty.

## Script Locations

All four standardization scripts live under the same directory:

```text
scripts/evaluation/data_prep/run_medical_cases_extraction.py
scripts/evaluation/data_prep/standardize_phenopackets.py
scripts/evaluation/data_prep/standardize_phenopackets_with_excluded.py
scripts/evaluation/data_prep/standardize_mygene2.py
```

`run_medical_cases_extraction.py` imports the extraction-based patient profile builder from the core package:

```python
from raresim.hpo_extraction import build_patient_profile
```

See [Patient Profile Construction](patient-profile-construction.md) for what that function does and how it differs from the example-patient builder of the same name.
