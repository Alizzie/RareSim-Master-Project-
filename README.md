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

## PhenoBERT setup

PhenoBERT is a CNN + BioBERT model purpose-built for HPO recognition.
It handles complex paraphrases that dictionary and synonym methods miss.

### Step 1: Clone the repo

```bash
cd /path/to/RareSim-Master-Project-
git clone https://github.com/EclipseCN/PhenoBERT.git
cd PhenoBERT
pip install -r requirements.txt
python setup.py install
```

### Step 2: Download pretrained models

Download from Google Drive:
https://drive.google.com/drive/folders/1jIqW19JJPzYuyUadxB5Mmfh-pWRiEopH

Then move files into the correct folders:

```bash
mkdir -p phenobert/embeddings phenobert/models
mv /path/to/downloads/embeddings/* phenobert/embeddings/
mv /path/to/downloads/models/* phenobert/models/
```

Expected folder structure after setup:

```
PhenoBERT/
└── phenobert/
    ├── models/
    │   ├── HPOModel_H/
    │   └── bert_model_max_triple.pkl
    └── embeddings/
        ├── biobert_v1.1_pubmed/
        └── fasttext_pubmed.bin
```

---

## Requirements

```bash
pip install transformers   # for biomedical_ner
# PhenoBERT has its own requirements — see setup above
```

All other methods (dictionary, synonyms) have no additional dependencies.

---

## Notes

- `hpo_synonyms.json` is built automatically by `src/build_shared_artifacts.py`
  from the HPO OWL ontology file. Re-run if HPO version is updated.