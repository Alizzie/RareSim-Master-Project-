# Patient Profile Construction

## Purpose

Patient profiles are the other object every similarity method reads from, alongside disease profiles.

A patient profile combines:

```text
Patient ID
Raw clinical text
Direct HPO terms
Propagated HPO terms
Excluded HPO terms
```

The patient profile schema is defined in:

```text
raresim/types/schemas.py
```

alongside `DiseaseProfile`. The dataclass is:

```python
PatientProfile
```

This page documents the general-purpose construction path used to turn a raw case (free text, hand-entered HPO terms, or a mix) into a `PatientProfile`. It is distinct from `example_patient.json`, which is a single static fixture defined in `raresim/core/config.py` and built during shared-artifact generation (see [Artifact File Reference and Runtime Loading](file-reference-and-runtime-loading.md)).

## PatientProfile Schema

Important fields:

```text
patient_id
    Case identifier.

raw_text
    The raw clinical description, if available.

hpo_terms
    Direct positive HPO terms, observed in the patient.

propagated_hpo_terms
    Direct terms plus ancestor HPO terms.

excluded_hpo_terms
    HPO terms explicitly documented as absent or negated.
```

This mirrors the disease side field-for-field: `hpo_terms` / `propagated_hpo_terms` / `negative_hpo_terms` on `DiseaseProfile` correspond to `hpo_terms` / `propagated_hpo_terms` / `excluded_hpo_terms` on `PatientProfile`. Keeping observed and excluded terms in separate fields, rather than one merged list, is deliberate: it is what prevents a symptom a patient explicitly does not have from being scored as if they did, and it is what makes explicit contradiction-scoring (see `set_jaccard_penalized` below) possible at all.

## Building a Patient Profile

**Important:** there are two distinct functions named `build_patient_profile()` in the codebase. They are not the same function defined twice — they take different inputs, live in different modules, and serve different purposes.

### 1. Extraction-based builder (general-purpose)

```python
raresim.hpo_extraction.ensemble.build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_labels: dict[str, str],
    methods: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]
```

Defined in:

```text
packages/raresim-core/src/raresim/hpo_extraction/ensemble.py
```

This is the general-purpose builder. It takes raw clinical text, runs one or more extraction methods over it, then propagates the extracted terms up the HPO ancestor hierarchy. It returns a `(patient, extracted_terms)` tuple:

- `patient` — dict with `patient_id`, `raw_text`, `hpo_terms`, `propagated_hpo_terms`, `methods_used`
- `extracted_terms` — list of dicts carrying full extraction provenance (which method found which term)

This is the function `run_medical_cases_extraction.py` imports and calls (see [Benchmark Dataset Standardization](benchmark-dataset-standardization.md)):

```python
from raresim.hpo_extraction import build_patient_profile
```

### 2. Example-patient builder (fixture-specific)

```python
raresim.build.build_shared_artifacts.build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_terms: list[str],
    hpo_labels: dict[str, str],
    hpo_ancestors: dict[str, set[str]],
) -> PatientProfile
```

Defined in:

```text
packages/raresim-core/src/raresim/build/build_shared_artifacts.py
```

## HPO Extraction Strategies

When a patient case only supplies raw clinical text, phenotype mentions are extracted and mapped to HPO identifiers. RareSim does not commit to a single extraction strategy; the following are supported and selectable:

```text
Dictionary-based matching
External extraction tools (e.g. FastHPOCR, PhenoBrain API)
Named entity recognition (NER)
GPT-based extraction (GPT-4o-mini)
Ensemble of the above
```

This is the same extraction machinery used by the Medical cases benchmark standardization step (`run_medical_cases_extraction.py`, selectable via `--methods`) — see [Benchmark Dataset Standardization](benchmark-dataset-standardization.md).

Dispatch lives in `raresim/hpo_extraction/ensemble.py`, in `extract_hpo_terms()`. All five method keys are confirmed:

```text
"dictionary"       -> extract_dictionary(raw_text, hpo_labels, skip_negated)      (exact label matching)
"biomedical_ner"   -> extract_biomedical_ner(raw_text, hpo_labels, skip_negated)  (d4data transformer NER)
"fast_hpo_cr"      -> extract_fast_hpo_cr(raw_text, hpo_labels, skip_negated)     (FastHPOCR morphological matching)
"chatgpt"          -> extract_chatgpt(raw_text, hpo_labels, skip_negated)         (GPT-4o-mini)
"phenobrain_api"   -> extract_phenobrain_api(raw_text, hpo_labels, skip_negated)  (PhenoBrain public API)
```

If `methods` is not supplied, it defaults to `["dictionary"]` only. Each selected branch's results are accumulated and passed through `deduplicate()` before being returned.

## Propagation

Once direct HPO terms exist (entered directly or extracted), propagation applies the same true-path-rule mechanism used on the disease side:

```python
propagate_hpo_terms()
```

from:

```text
raresim/ontology/hpo_utils.py
```

behavior controlled by the same flag used for disease profiles:

```python
APPLY_TRUE_PATH_RULE = True
```

in `raresim/core/config.py`. When enabled, every direct patient HPO term is expanded with its ancestor terms from `hpo_ancestors.json`, producing `propagated_hpo_terms`.

## Observed vs. Excluded Terms

Findings explicitly documented as absent or negated are never merged into `hpo_terms`. They are stored separately in `excluded_hpo_terms`.

This negative information is preserved regardless of extraction method, but it only changes a ranking for similarity methods that are built to look at it. Currently, only one set-based variant does:

```text
set_jaccard_penalized
```

which applies an explicit penalty when a patient's `excluded_hpo_terms` overlap with a disease's positive `hpo_terms`, or vice versa (using the disease-side `negative_hpo_terms` field from [Disease Profile Construction](disease-profile-construction.md)). All other similarity families currently retain `excluded_hpo_terms` on the profile but do not incorporate it into scoring.

## Relationship to `example_patient.json`

`example_patient.json` (documented in [Artifact File Reference and Runtime Loading](file-reference-and-runtime-loading.md)) is a single fixed patient defined by `EXAMPLE_PATIENT` in `raresim/core/config.py`, built once during shared-artifact generation for demo/testing purposes. It does not carry an `excluded_hpo_terms` field in its current definition. Real patient cases — whether a live query or a benchmark case — go through `build_patient_profile()` as described on this page, not through the example-patient fixture path.
