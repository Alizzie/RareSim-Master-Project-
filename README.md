# RareSim Master Project

# 1. Setup Project

## Download Ontologies
The ontologies are not stored direclty as it takes too much storage space.
To download the ontologies to you local file system, you can use the script in ontologies/model/load_models_to_local.


## Neo4J
For the knowledge graph, we are utilizing Neo4J. Start it via Docker

```
docker run -d --name neo4j-raresim \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/raresim123 \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  neo4j:5
```

## Phenotype Extraction Setup

The phenotype extraction pipeline (`src/shared/phenotype.py`) supports multiple methods. Some require additional setup:

### Dictionary
No setup required. Uses regex matching against HPO labels.

### Biomedical NER (d4data)
Requires `transformers` and `torch` — already in `requirements.txt`. The model (`d4data/biomedical-ner-all`) will be downloaded automatically from HuggingFace on first use.

### FastHPOCR
Morphological HPO concept recognition (recommended):

```bash
# Clone into src/
git clone https://github.com/tudorgroza/fast_hpo_cr.git src/fast_hpo_cr

# Download hp.obo (FastHPOCR uses OBO format, separate from hpo.owl)
wget https://purl.obolibrary.org/obo/hp.obo -O ontologies/model/hp.obo
```

The index will be built automatically on first use (~12 min) and cached to `outputs/fast_hpo_cr_index/`.

### ChatGPT Extraction
Requires an OpenAI API key. Add to your `.env` file:
OPENAI_API_KEY=sk-...

### PhenoBrain API
No API key required. Uses the public PhenoBrain endpoint. Requires `pip install requests` (already in `requirements.txt`).

# 2. How to Run
```
pip install -e .
```

## Step 1 — download raw ontology files
```
python ontologies/load_models_to_local.py
```

## Step 2 — process them into shared artifacts
```
python -m build_shared_artifacts
```

## Step 3 — run the app
```
python -m gui.app --defaults
```