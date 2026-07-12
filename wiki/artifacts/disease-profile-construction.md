# Disease Profile Construction

## Purpose

Disease profiles are the main objects used later by similarity pipelines.

A disease profile combines:

```text
Disease ID
Disease label
Disease metadata
Direct HPO terms
Propagated HPO terms
Descriptions
Source IDs
Aliases
Term provenance
Negative phenotype assertions
```

Disease profile construction is implemented in:

```text
raresim/ontology/disease_profiles.py
```

The main function is:

```python
build_canonical_disease_profiles()
```

## Inputs

The profile builder receives:

```text
Merged phenotype annotation records
Term provenance
Negative phenotype assertions
HPO labels
HPO ancestors
ORDO metadata
MONDO metadata
HOOM metadata
ORPHA mapping index
True-path-rule flag
```

These inputs come from earlier artifact-building steps.

## Canonical Disease Profiles

The output is:

```text
canonical_profiles
alias_to_canonical
```

The canonical profiles are saved as:

```text
canonical_disease_profiles.json
```

The alias mapping is saved as:

```text
alias_to_canonical.json
```

A canonical profile is preferably keyed by an ORPHA ID when a reliable ORPHA mapping exists.

If no reliable ORPHA mapping exists, the normalized original disease ID is kept.

## Processing One Annotation Record

For each phenotype annotation record, the builder:

1. Normalizes the disease ID.
2. Normalizes the HPO ID.
3. Skips the record if the HPO ID is invalid or unknown.
4. Resolves the disease ID to ORPHA when a reliable mapping exists.
5. Gets or creates the canonical disease profile.
6. Adds the HPO term to the profile.
7. Stores the original source ID.
8. Stores frequency codes as metadata.
9. Copies term provenance.
10. Adds negative HPO terms if available.
11. Marks the profile as canonicalized if the ID changed.

Example:

```text
Raw annotation:
    MONDO:0000437 -> HP:0001250

Mapping:
    MONDO:0000437 -> ORPHA:102002

Canonical profile:
    ORPHA:102002 contains HP:0001250

Alias map:
    MONDO:0000437 -> ORPHA:102002
```

## Phenotype Annotation Merging

Before profiles are built, annotations are merged in:

```text
raresim/ontology/phenotype_merge.py
```

The main function is:

```python
merge_phenotype_annotation_records()
```

It groups records by:

```text
(database_id, hpo_id)
```

For each disease-HPO pair, it:

1. Normalizes frequency values.
2. Separates positive annotations from negative assertions.
3. Selects the best positive annotation.
4. Stores provenance information.
5. Stores excluded/negative HPO terms separately.

Source priority:

```text
HPOA > ORPHADATA_PRODUCT4 > HOOM > MONARCH
```

Frequency values are normalized into values such as:

```text
VERY_RARE
OCCASIONAL
FREQUENT
VERY_FREQUENT
OBLIGATE
EXCLUDED
```

Negative assertions are detected when:

```text
qualifier == "NOT"
```

or when normalized frequency is:

```text
EXCLUDED
```

Negative terms are not added to positive HPO terms. They are stored separately in:

```text
negative_terms_by_disease.json
```

## Term Provenance

The merge step records where each disease-HPO annotation came from.

The artifact is:

```text
term_provenance.json
```

It stores:

```text
selected_source
selected_frequency
all_sources
all_frequencies
had_negative_assertion
excluded_from_positive_annotations
```

This is useful for debugging and later explanation.

## True-Path Rule

Disease profiles store two HPO term sets:

```text
hpo_terms
    Direct positive HPO annotations.

propagated_hpo_terms
    Direct HPO annotations plus ancestor HPO terms.
```

The true-path rule is applied using:

```python
propagate_hpo_terms()
```

from:

```text
raresim/ontology/hpo_utils.py
```

The behavior is controlled by:

```python
APPLY_TRUE_PATH_RULE = True
```

in:

```text
raresim/core/config.py
```

When enabled, every direct HPO term is expanded with its ancestor terms from:

```text
hpo_ancestors.json
```

## DiseaseProfile Schema

The disease profile schema is defined in:

```text
raresim/types/schemas.py
```

The dataclass is:

```python
DiseaseProfile
```

Important fields:

```text
disease_id
    The disease profile ID. Preferably ORPHA when mapping is available.

label
    Display label for the disease.

profile_type
    Disease/category type, usually from ORDO metadata.

hpo_terms
    Direct positive HPO annotations.

propagated_hpo_terms
    Direct annotations plus ancestor HPO terms.

ordo_label / ordo_description
    Label and description from ORDO.

mondo_label / mondo_description
    Label and description from MONDO.

hoom_label / hoom_description
    Label and description from HOOM.

merged_description
    Selected disease description.

source_ids
    Original source IDs and metadata used to build the profile.

aliases
    Equivalent disease IDs, if available.

category_source_id
    ID used for category metadata, if available.

canonicalized_to_orpha
    True when the original ID was mapped to a different ORPHA ID.

term_provenance
    Source and frequency information for selected HPO annotations.

negative_hpo_terms
    HPO terms explicitly excluded for the disease.
```

## Metadata Added to Profiles

Metadata from ORDO, MONDO, and HOOM is added to disease profiles.

This can include:

```text
Labels
Descriptions
Profile type
Source IDs
Local ontology IDs
```

The merged disease description is selected from available source descriptions.

Current priority:

```text
ORDO description
MONDO description
HOOM description
```

The first available non-empty description is used.

## Expanded Alias Profiles

In the current build script, canonical profiles are built first, alias profiles are expanded second, and then canonical and expanded profile dictionaries are filtered separately.

Function:

```python
expand_alias_profiles()
```

This produces:

```text
disease_profiles.json
```

Difference between the two profile files:

```text
canonical_disease_profiles.json
    One profile per canonical disease concept.

disease_profiles.json
    Canonical profiles plus alias-keyed copies.
```

Example:

```text
canonical_disease_profiles.json:
    ORPHA:102002 -> profile

alias_to_canonical.json:
    MONDO:0000437 -> ORPHA:102002
    OMIM:123456 -> ORPHA:102002

disease_profiles.json:
    ORPHA:102002 -> profile
    MONDO:0000437 -> copied profile
    OMIM:123456 -> copied profile
```

The expanded file is useful when an input dataset uses OMIM, MONDO, DECIPHER, or another alias instead of the canonical disease ID.

## Filtering Profiles

After profiles are built and expanded, the build script filters disease profiles.

In the current build script, a profile is kept if it has:

```text
at least one HPO term
```

or:

```text
a non-empty merged description
```

Profiles with neither HPO terms nor descriptions are removed.

The filtering statistics are saved as:

```text
canonical_filter_stats.json
expanded_filter_stats.json
```

Important note:

```text
MIN_DISEASE_HPO_TERMS is defined in config.py, but the current filter shown in build_shared_artifacts.py does not directly apply this threshold. The current filter uses the condition: has HPO terms OR has description.
```
