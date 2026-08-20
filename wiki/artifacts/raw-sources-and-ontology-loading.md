# Raw Sources and Ontology Loading

## Purpose

This page explains how raw ontology and source files are loaded before disease profiles are built.

Raw files are stored under:

```text
data/ontologies/
```

The relevant paths are defined in:

```text
raresim/utils/paths.py
```

## Downloading Raw Source Files

Before artifacts can be built, the raw ontology and annotation files must exist locally. On a fresh setup, download them with:

```bash
python -m raresim.build.load_ontologies_to_local
```

This script is defined in:

```text
raresim/build/load_ontologies_to_local.py
```

Its purpose is to download the external ontology and annotation files into the local ontology directory:

```text
data/ontologies/
```

Internally, the script uses:

```python
ONTOLOGY_DIR
```

from:

```text
raresim/utils/paths.py
```

The script downloads ontology, annotation, and auxiliary files used by RareSim. Most are used by build_shared_artifacts.py. The hpo.obo file is mainly used by the FastHPOCR extraction/indexing workflow, not directly by build_shared_artifacts.py.

| Source key | Downloaded local file |
|---|---|
| `hpo` | `hpo.owl` |
| `hpo_obo` | `hpo.obo` |
| `mondo_rare` | `mondo_rare.owl` |
| `ordo` | `ordo.owl` |
| `hoom` | `hoom.owl` |
| `phenotype_hpoa` | `phenotype.hpoa` |
| `orphadata_product4` | `en_product4_HPO.xml` |
| `monarch_disease_hpo` | `disease_to_phenotypic_feature_association.all.tsv.gz` |

If a file already exists, it is not downloaded again. This means the step is required for a fresh setup, but safe to rerun later.
Note: hpo.obo is downloaded here, but the shared artifact build script uses hpo.owl. hpo.obo is needed by the FastHPOCR extraction/indexing workflow.

Important distinction:

```text
load_ontologies_to_local.py
    Downloads raw source files.

build_shared_artifacts.py
    Parses raw source files and builds JSON artifacts.
```

The required setup order for a fresh environment is:

```bash
python -m raresim.build.load_ontologies_to_local
python -m raresim.build.build_shared_artifacts
```


## Raw Input Files

RareSim uses the following raw files.

### HPO ontology

```text
hpo.owl
```

Used to extract:

- HPO labels
- HPO parent relations
- HPO ancestors

Generated artifacts:

```text
hpo_labels.json
hpo_parents.json
hpo_ancestors.json
```

### ORDO ontology

```text
ordo.owl
```

Used to extract:

- ORDO disease metadata
- ORPHA disease IDs
- Disease labels
- Disease descriptions
- Disease profile types
- Disease parent relations
- Disease ancestor paths

Generated artifacts:

```text
disease_metadata_index.json
disease_parents.json
disease_ancestors.json
```

### MONDO ontology

```text
mondo_rare.owl
```

Used to extract:

- MONDO metadata
- MONDO labels
- MONDO descriptions
- MONDO xrefs
- MONDO exact matches
- MONDO-to-ORPHA mappings

### HOOM ontology

```text
hoom.owl
```

Used in the current build to extract disease-HPO annotations and frequency information. The codebase also has support for HOOM metadata loading, but the current shared-artifact build passes an empty HOOM metadata dictionary.

### HPOA annotations

```text
phenotype.hpoa
```

Used to extract disease-to-HPO annotations.

### Orphadata Product 4

```text
en_product4_HPO.xml
```

Used to extract Orphanet disease-to-HPO annotations and phenotype frequencies.

### Monarch disease-HPO associations

```text
disease_to_phenotypic_feature_association.all.tsv.gz
```

Used to extract disease-to-HPO annotations from Monarch.

## Loader Module

Raw source parsing is implemented in:

```text
raresim/ontology/loaders.py
```

Important loader functions:

```text
load_hpo_owl()
load_hpoa_annotations()
load_hoom_hpo_annotations()
load_orphadata_product4_annotations()
load_monarch_disease_hpo_annotations()
load_ordo_metadata()
load_mondo_metadata()
load_ordo_parents()
```

## HPO Loading

The function:

```python
load_hpo_owl()
```

parses `hpo.owl`.

It extracts two objects:

```text
hpo_labels
    HPO ID -> readable HPO label

hpo_parents
    HPO ID -> direct parent HPO IDs
```

Example:

```json
{
  "HP:0001250": "Seizure"
}
```

The outputs are saved as:

```text
hpo_labels.json
hpo_parents.json
```

Then HPO ancestors are computed using:

```python
compute_ancestors()
```

from:

```text
raresim/ontology/hpo_utils.py
```

This produces:

```text
hpo_ancestors.json
```

## Disease Metadata Loading

Disease metadata is loaded mainly from ORDO and MONDO.

The generic metadata loader extracts:

```text
uri
normalized_id
label
description
xrefs
exact_matches
profile_type
```

This metadata is later used to:

- Build disease profiles
- Choose disease labels
- Attach descriptions
- Create mapping indexes
- Prepare disease category metadata

## Phenotype Annotation Loading

Disease-HPO annotations are loaded from multiple sources:

```text
HPOA
HOOM
Orphadata Product 4
Monarch
```

All loaders return records in a shared format:

```json
{
  "database_id": "ORPHA:123",
  "disease_name": "Disease name",
  "qualifier": "",
  "hpo_id": "HP:0000001",
  "frequency_code": "Frequent",
  "source": "ORPHADATA_PRODUCT4"
}
```

Using a common record format makes it possible to merge annotations from different sources later.

## ORDO Disease Parent Loading

Disease hierarchy is loaded with:

```python
load_ordo_parents()
```

This parses direct ORDO `subClassOf` relations.

It returns:

```text
ORPHA disease/category ID -> direct ORDO parent IDs
```

Example structure:

```json
{
  "ORPHA:123": ["ORPHA:456"]
}
```

This becomes:

```text
disease_parents.json
```

Then ordered ancestor chains are computed and saved as:

```text
disease_ancestors.json
```
