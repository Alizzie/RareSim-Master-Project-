# RareSim

A rare disease similarity pipeline for phenotype-driven diagnosis support.
Given a patient's HPO terms or clinical text, RareSim ranks diseases by phenotypic similarity using multiple methods — semantic IC-based, set-based, TF-IDF, transformer, and LLM.

---

## Overview

```
Patient HPO terms / clinical text
        │
        ▼
  HPO Extraction          ← dictionary, NER, FastHPOCR, GPT, PhenoBrain
        │
        ▼
  Similarity Pipeline     ← semantic (Resnik/Lin/JC), set-based, TF-IDF, transformer, LLM
        │
        ▼
  Ranked Disease Results  ← scored against ~10,000 rare diseases
```

Disease knowledge is built from four ontologies: **HP**, **ORDO**, **MONDO**, and **HOOM**, merged and canonicalized to ORPHA identifiers.

---

## Repository Structure

```
RareSim/
  packages/
    raresim-core/          ← installable Python package (core logic)
    raresim-api/           ← FastAPI backend (wraps raresim-core)
    raresim-frontend/      ← Vue 3 web interface

  scripts/
    setup/                 ← one-time setup: ontologies, artifacts, third-party tools
    evaluation/            ← evaluate pipeline performance
    validation_tools/      ← run and compare existing tools (LIRICAL etc.)
    experiments/           ← ad-hoc experiments
    analysis/              ← analyse outputs, automated reporting

  tests/
    unit/
    integration/
    validation_tool/
    evaluation/

  data/                    ← gitignored large files (see data/README.md)
    ontologies/            ← hp.owl, ordo.owl, mondo.owl, hoom.owl
    datasets/              ← HMS.json, MME.json, LIRICAL.json, phenopackets/

  outputs/                 ← gitignored, generated at runtime
    artifacts/             ← precomputed profiles, IC, ancestors, labels
    transformer/
    semantic/
    evaluation/
    validation/
    gui/

  third_party/             ← externally cloned tools (gitignored)
    fast_hpo_cr/

  docs/
    notebooks/

  pyproject.toml           ← root: dev tooling + uv workspace
  .env                     ← local paths (not committed)
  README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Node.js 18+ (frontend only)
- `uv` (recommended) or `pip`

### 2. Clone the repo

```bash
git clone https://github.com/your-org/RareSim.git
cd RareSim
```

### 3. Set up environment

```bash
# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install all packages (uv workspace)
uv sync

# or with pip
pip install -e packages/raresim-core
pip install -e packages/raresim-api
```

### 4. Configure paths

Create a `.env` file at the repo root:

```bash
RARESIM_ROOT=/path/to/RareSim
OPENAI_API_KEY=sk-...        # optional — only needed for GPT extraction
```

### 5. Set up third-party tools

```bash
python scripts/setup/setup_third_party.py
```

This clones [FastHPOCR](https://github.com/tudorgroza/fast_hpo_cr) into `third_party/`. Skips anything already cloned.

### 6. Download ontologies

Download the following files into `data/ontologies/` (see `data/README.md` for exact versions and sources):

```bash
python scripts/setup/load_ontologies_to_local.py
```

| File | Source |
|------|--------|
| `hp.owl` | [hpo.jax.org](https://hpo.jax.org) |
| `ordo.owl` | [BioPortal](https://bioportal.bioontology.org/ontologies/ORDO) |
| `mondo.owl` | [GitHub](https://github.com/monarch-initiative/mondo) |
| `hoom.owl` | [BioPortal](https://bioportal.bioontology.org/ontologies/HOOM) |
| `phenotype.hpoa` | [hpo.jax.org](https://hpo.jax.org) |
| `en_product4_HPO.xml` | [Orphadata](https://www.orphadata.com) |
| `disease_to_phenotypic_feature_association.all.tsv.gz` | [Monarch](https://monarch-initiative.github.io) |

### 7. Build shared artifacts

```bash
python scripts/setup/build_shared_artifacts.py
```

Generates precomputed files in `outputs/artifacts/`:
- `canonical_disease_profiles.json`
- `hpo_labels.json`
- `hpo_ancestors.json`
- `information_content.json`
- `alias_to_canonical.json`

### 8. Standardize phenopackets (optional)

Only needed if you are using phenopacket datasets for evaluation.

```bash
python scripts/setup/standardize_phenopackets.py
```

Converts raw phenopackets from `data/datasets/phenopackets/raw/` into a standardized `[[HP terms], [disease codes]]` format, saved per release folder under `data/datasets/phenopackets/standardized_to_json/`.


### 9. Run the pipeline

**Terminal interface:**
```bash
# from HPO terms
python scripts/raresim_cli/app.py --hpo HP:0001251,HP:0000545

# from clinical text
python scripts/run_pipeline.py --text "Patient with cerebellar ataxia and macrocephaly."

# from patient file
python scripts/run_pipeline.py --patient data/patient_profiles/example_patient.json

# all methods, example patient
python scripts/run_pipeline.py --defaults
```

**Web interface:**
```bash
# terminal 1 — backend
uvicorn raresim_api.main:app --reload --port 8000

# terminal 2 — frontend
cd packages/raresim-frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Similarity Methods

| Method | Type | Description |
|--------|------|-------------|
| `semantic_resnik_bma` | Semantic | Resnik IC-based Best Match Average |
| `semantic_lin_bma` | Semantic | Lin normalized IC similarity |
| `semantic_jiang_conrath_bma` | Semantic | Jiang-Conrath distance-based |
| `set_cosine` | Set-based | Cosine similarity over HPO term vectors |
| `set_jaccard` | Set-based | Jaccard index |
| `set_dice` | Set-based | Sørensen-Dice coefficient |
| `set_overlap` | Set-based | Szymkiewicz-Simpson overlap |
| `tfidf` | TF-IDF | Term frequency-inverse disease frequency |
| `transformer` | Embedding | Sentence transformer cosine similarity |
| `llm` | LLM | GPT-based disease ranking |

---

## HPO Extraction Methods

| Method | Description |
|--------|-------------|
| `dictionary` | Exact HPO label matching (fast baseline) |
| `biomedical_ner` | d4data/biomedical-ner-all transformer NER |
| `fast_hpo_cr` | FastHPOCR morphological token cluster matching |
| `chatgpt` | GPT-4o-mini prompted extraction |
| `phenobrain_api` | PhenoBrain public BERT-based API |


The phenotype extraction pipeline supports multiple methods. Some require additional setup:

### Dictionary
No setup required. Uses regex matching against HPO labels.

### Biomedical NER (d4data)
Requires `transformers` and `torch` — already in `requirements.txt`. The model (`d4data/biomedical-ner-all`) will be downloaded automatically from HuggingFace on first use.

### FastHPOCR
Morphological HPO concept recognition (recommended):
The index will be built automatically on first use (~12 min) and cached to `outputs/fast_hpo_cr_index/`.

### ChatGPT Extraction
Requires an OpenAI API key. Add to your `.env` file:
OPENAI_API_KEY=sk-...

### PhenoBrain API
No API key required. Uses the public PhenoBrain endpoint. Requires `pip install requests` (already in `requirements.txt`).

---

## Development

### Running tests

```bash
pytest tests/
```

### Code style

```bash
ruff check .
ruff format .
```

### Adding a new similarity method

1. Create `packages/raresim-core/src/raresim/similarity_methods/<name>/`
2. Add `methods.py` and `pipeline.py` following existing patterns
3. Register the method name in `scripts/run_pipeline.py`

---

## Package Architecture

```
raresim-core            ← pure logic, no HTTP
    │  Python import
raresim-api             ← FastAPI, wraps raresim-core
    │  HTTP /api/*
raresim-frontend        ← Vue 3, calls raresim-api
```

`raresim-core` has zero knowledge of the API or frontend. Scripts and tests import directly from `raresim-core`.

---

## Data Sources

See [`data/README.md`](data/README.md) for full provenance, versions, and download instructions.

---

## License

[MIT](LICENSE)