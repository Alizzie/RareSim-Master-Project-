# Artifact File Reference and Runtime Loading

## Purpose

This page explains the generated artifact files and how they are loaded later by RareSim.

Artifacts are saved under:

```text
outputs/artifacts/
```

The build script saves them using:

```python
save_json()
```

from:

```text
raresim/utils/io.py
```

## Generated Artifact Files

### `canonical_disease_profiles.json`

Main cleaned disease profile file.

Contains disease profiles keyed by canonical disease IDs.

Usually, canonical IDs are ORPHA IDs when reliable mappings exist.

This is the default disease profile file used at runtime because PipelineConfig.use_canonical_profiles defaults to True.

Use this file when duplicate aliases should not be counted as separate disease concepts.

### `disease_profiles.json`

Expanded disease profile file.

Contains canonical disease profiles plus alias-keyed copies.

Useful when input datasets or external tools use disease IDs such as:

```text
OMIM
MONDO
DECIPHER
DOID
```

instead of canonical ORPHA IDs.

### `hpo_labels.json`

Maps HPO IDs to readable labels.

Example:

```json
{
  "HP:0001250": "Seizure"
}
```

Used to validate HPO terms and display readable phenotype names.

### `hpo_parents.json`

Stores direct HPO parent relations.

Format:

```text
HPO ID -> direct parent HPO IDs
```

Used to compute HPO ancestors.

### `hpo_ancestors.json`

Stores all ancestor terms for each HPO term.

Format:

```text
HPO ID -> all ancestor HPO IDs
```

Used for:

- True-path propagation
- Ontology-aware similarity
- Patient and disease term expansion

### `disease_parents.json`

Stores direct ORDO disease/category parent relations.

Format:

```text
ORPHA ID -> direct parent ORPHA IDs
```

### `disease_ancestors.json`

Stores ordered ORDO ancestor paths.

Format:

```text
ORPHA ID -> [root, ..., immediate parent]
```

Used for disease category paths and hierarchy-aware interpretation.

### `disease_metadata_index.json`

Stores readable metadata for diseases and ORDO categories.

Typical fields:

```text
label
profile_type
```

Used later to display readable disease/category names.

### `orpha_mapping_index.json`

Maps non-ORPHA IDs to ORPHA IDs when reliable mappings exist.

Examples:

```text
OMIM:301310 -> ORPHA:123
MONDO:0000437 -> ORPHA:102002
```

Used during canonical disease profile construction.

### `alias_to_canonical.json`

Maps aliases to canonical disease IDs.

Example:

```json
{
  "MONDO:0000437": "ORPHA:102002",
  "OMIM:123456": "ORPHA:102002"
}
```

Used to resolve source-specific IDs to canonical profiles.

### `term_frequencies.json`

Stores how many canonical disease profiles contain each HPO term.

Computed from canonical disease profiles.

By default, propagated HPO terms are counted.

### `information_content.json`

Stores information content values for HPO terms.

Computed using:

```text
IC(term) = -log(freq(term) / total_diseases)
```

Interpretation:

```text
Common HPO terms:
    low IC

Rare/specific HPO terms:
    high IC
```

Used later by semantic similarity methods.

### `term_provenance.json`

Stores provenance information for disease-HPO annotations.

Fields include:

```text
selected_source
selected_frequency
all_sources
all_frequencies
had_negative_assertion
excluded_from_positive_annotations
```

Useful for debugging and explanation.

### `negative_terms_by_disease.json`

Stores explicitly excluded HPO terms for diseases.

These terms are not treated as positive phenotype annotations.

### `annotation_source_counts.json`

Stores summary counts for loaded annotation sources.

Useful for checking that loaders are working correctly.

### `canonical_filter_stats.json`

Stores filtering statistics for canonical profiles.

### `expanded_filter_stats.json`

Stores filtering statistics for expanded alias profiles.

### `example_patient.json`

Stores an example patient profile.

The example patient is configured in:

```text
raresim/core/config.py
```

The example patient terms are normalized and propagated before saving.

## Information Content Computation

Information content is computed in:

```text
raresim/ontology/ic.py
```

The function:

```python
compute_term_frequencies()
```

counts how many disease profiles contain each HPO term.

By default, it uses:

```text
propagated_hpo_terms
```

Then:

```python
compute_information_content()
```

computes:

```text
IC(term) = -log(freq(term) / total_diseases)
```

These outputs are built from canonical profiles, not expanded alias profiles. This avoids counting alias copies as separate diseases.

## Example Patient

The example patient is defined in:

```text
raresim/core/config.py
```

Example:

```python
EXAMPLE_PATIENT = {
    "patient_id": "patient_001",
    "raw_text": "Patient with developmental delay, cerebellar ataxia, and anemia.",
    "hpo_terms": ["HP:0001263", "HP:0002470", "HP:0001903"],
}
```

During artifact generation:

1. HPO terms are normalized.
2. Terms are checked against `hpo_labels`.
3. Ancestor HPO terms are added using `hpo_ancestors`.
4. The result is saved as `example_patient.json`.

## Runtime Loading with AppContext

At runtime, pipelines load shared artifacts through:

```python
AppContext.load(patient, use_canonical_profiles=True)
```

Defined in:

```text
raresim/core/context.py
```

If:

```python
use_canonical_profiles=True
```

then the context loads:

```text
canonical_disease_profiles.json
```

If:

```python
use_canonical_profiles=False
```

then it loads:

```text
disease_profiles.json
```

`AppContext` also loads:

```text
hpo_labels.json
information_content.json
hpo_ancestors.json
disease_ancestors.json
disease_metadata_index.json
hpo_parents.json
alias_to_canonical.json
```

The loaded context contains:

```text
disease_profiles
hpo_labels
ic_values
ancestors
disease_ancestors
disease_metadata_index
hpo_parents
alias_to_canonical
app_metadata
```

It also checks whether all patient HPO terms exist in `hpo_labels.json`.

Missing patient terms are stored in:

```text
app_metadata.unfound_patient_terms
```