# Shared Artifact Overview

## Purpose

Before any similarity method runs, RareSim builds a set of shared JSON artifacts.

These artifacts contain cleaned and standardized ontology data, disease profiles, HPO labels, disease mappings, metadata, ancestor relations, and information content values.

The purpose is to avoid parsing raw ontology files every time a method runs.

The overall workflow is:

```text
Raw ontology and annotation files
    ↓
src/raresim/build/build_shared_artifacts.py
    ↓
outputs/artifacts/*.json
    ↓
AppContext.load()
    ↓
Similarity pipelines
```

Similarity methods do not directly parse files such as:

```text
hpo.owl
ordo.owl
mondo_rare.owl
hoom.owl
phenotype.hpoa
en_product4_HPO.xml
disease_to_phenotypic_feature_association.all.tsv.gz
```

Instead, they load prepared JSON artifacts through `AppContext`.

## Main Entry Point

The main artifact generation script is:

```text
src/raresim/build/build_shared_artifacts.py
```

This script orchestrates the preprocessing workflow.

It performs these steps:

1. Load HPO ontology.
2. Compute HPO ancestors.
3. Load disease-HPO phenotype annotations from multiple sources.
4. Merge and deduplicate phenotype annotations.
5. Load disease metadata from ORDO and MONDO.
6. Build disease ID mappings.
7. Build canonical disease profiles.
8. Expand canonical profiles to aliases.
9. Filter invalid or empty profiles.
10. Build disease hierarchy artifacts.
11. Compute HPO term frequencies and information content.
12. Build an example patient profile.
13. Save all generated artifacts as JSON files.

The script coordinates the workflow, but most parsing and transformation logic is implemented in:

```text
raresim/ontology/
raresim/utils/
raresim/types/
raresim/core/
```

## Build-Time vs Runtime

The artifact workflow has two phases.

### Build-time

At build time, raw ontology and annotation files are parsed and converted into JSON artifacts.

```text
Raw source files
    ↓
build_shared_artifacts.py
    ↓
outputs/artifacts/
```

### Runtime

At runtime, similarity pipelines load the generated JSON files through `AppContext`.

```text
outputs/artifacts/
    ↓
AppContext.load()
    ↓
Similarity methods
```

This separation keeps the similarity methods cleaner and ensures that all methods use the same shared data.

## Main Files

### `src/raresim/build/build_shared_artifacts.py`

Main artifact generation script.

Responsible for coordinating ontology loading, phenotype annotation merging, disease profile construction, disease hierarchy creation, information content computation, and JSON saving.

### `raresim/utils/paths.py`

Defines all important input and output paths.

Raw ontology files are expected under:

```text
data/ontologies/
```

Generated artifacts are saved under:

```text
outputs/artifacts/
```

The project root is read from:

```text
RARESIM_ROOT
```

### `raresim/utils/io.py`

Contains JSON loading and saving helpers.

Important functions:

```python
load_json(input_path)
save_json(data, output_path)
```

### `raresim/core/context.py`

Defines `AppContext`.

`AppContext` loads generated artifacts at runtime and makes them available to pipelines.

### `raresim/ontology/loaders.py`

Parses raw ontology and annotation files.

### `raresim/ontology/disease_profiles.py`

Builds canonical and expanded disease profiles.

### `raresim/utils/normalizers.py`

Normalizes HPO and disease IDs.

### `raresim/utils/mapping_utils.py`

Builds disease ID mappings, especially mappings to ORPHA IDs.

### `raresim/ontology/phenotype_merge.py`

Merges duplicate disease-HPO annotations and handles negative assertions.

### `raresim/ontology/hpo_utils.py`

Computes HPO ancestors and applies true-path propagation.

### `raresim/ontology/disease_ancestors.py`

Builds ORDO disease ancestor chains.

### `raresim/ontology/disease_category.py`

Builds readable disease category metadata used later in result interpretation.

### `raresim/ontology/ic.py`

Computes HPO term frequencies and information content values.

### `raresim/types/schemas.py`

Defines `DiseaseProfile` and `PatientProfile`.
