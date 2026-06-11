# Data

All large files are gitignored. See sections below to download them.

## Ontologies (`data/ontologies/`)

Run to download:

```bash
python scripts/setup/load_ontologies.py
```

### Included Ontologies:

| File | Ontology | Version | Release Date | Notes | Source |
|------|----------|---------|--------------|-------|--------|
| hp.owl | Human Phenotype Ontology (HP) | v16.2.2026 | 16.02.2026 | — | [hpo.jax.org](https://hpo.jax.org) · [GitHub](https://github.com/obophenotype/human-phenotype-ontology/) |
| ordo.owl | Orphanet Rare Disease Ontology (ORDO) | v4.8 | 20.06.2013 | Release date does not match upload date of 10.12.2025 | [BioPortal](https://bioportal.bioontology.org/ontologies/ORDO) · [Mondo Ingest](https://monarch-initiative.github.io/mondo-ingest/sources/ordo/) |
| hoom.owl | HPO-ORDO Ontological Module (HOOM) | v2.5 | 08.03.2018 | Release date does not match upload date of 10.12.2025 | [BioPortal](https://bioportal.bioontology.org/ontologies/HOOM) |
| mondo.owl | Mondo Disease Ontology | no version stated | — | Downloaded 11.03.2026 | [GitHub](https://github.com/monarch-initiative/mondo) |


## Datasets (`data/datasets/phenobrain_testdata`)

Download: https://zenodo.org/records/10774650

## Free Txt (`data/datasets/free_text`)

Ali Khan, Suleman (2025). "Can LLMs Help with Rare Diseases?" Bachelor's Theses. University of Zurich (UZH)

## Phenopackets (`data/datasets/phenopackets/`)

Source: https://github.com/monarch-initiative/phenopacket-store/releases
Version used: 0.1.26, 0.1.27

Download and extract into `data/datasets/phenopackets/`, then run:
\```bash
python scripts/setup/standardize_phenopackets.py
\```