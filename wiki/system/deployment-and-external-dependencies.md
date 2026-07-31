# Deployment and External Dependencies

## Purpose

This page covers how RareSim's processes are actually deployed locally, and every external system it talks to.

## Process Model

Three application layers run as three separate processes in development:

```text
Vue frontend      Vite development server
FastAPI backend    Own process
RareSim core       Invoked in-process by the backend on every request,
                   NOT a separate service
```

The core package is also invoked **directly**, outside the backend entirely, by the batch runners and evaluation scripts (see [Evaluation](/evaluation/workflow-overview)) and the [CLI](/system/cli). So "the architecture" isn't one long-running service — it's a shared library used by two different kinds of caller: an interactive API process, and a set of offline batch scripts.

## Local Execution Dependencies

Two external validation tools are invoked as **local command-line processes** via `subprocess`, despite both also having a live web presence for other purposes:

### LIRICAL

`run_lirical.py` shells out to a local LIRICAL JAR:

```bash
java -jar lirical-cli.jar prioritize -p <HPO IDs> --use-orphanet -o <dir>
```

Requires a local Java installation, the JAR file itself, and a local LIRICAL data directory, all passed as command-line arguments. Results are read back from the TSV file LIRICAL writes to disk — not from any return value or API response.

### Phenomizer

`run_phenomiser.py` follows the identical pattern:

```bash
java -Xmx<heap> -jar phenomiser-cli.jar query -hpo <hp.obo> -da <phenotype.hpoa>
```

Requires local copies of the HPO ontology and annotation files. Results are parsed from a text output file.

## External Network Dependencies

### Hugging Face model downloads

The transformer pipeline's five default models are Hugging Face model identifiers:

```text
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
emilyalsentzer/Bio_ClinicalBERT
sentence-transformers/all-MiniLM-L6-v2
cambridgeltl/SapBERT-from-PubMedBERT-fulltext
dmis-lab/biobert-v1.1
```

The LLM pipeline uses `mistralai/Mistral-7B-Instruct-v0.2` the same way.

Loading goes through two different libraries depending on model type:

```text
sentence-transformers models   -> SentenceTransformer class
Everything else                 -> transformers' AutoTokenizer/AutoModel,
                                    with manual mean-pooling over token
                                    embeddings
```

Both libraries download and cache weights automatically via `from_pretrained()` — Hugging Face is a required dependency the first time each model is used.

Device placement is automatic rather than per-method-configured: a single `get_device()` call routes to CUDA whenever `torch.cuda.is_available()`, and every transformer call uses whatever that returns.

### GPT-based extraction

Authentication: `OPENAI_API_KEY` environment variable, loaded from a `.env` file.

### PhenoBrain API (NEEDS TO BE UPDATED IF WE RUN LOCALLY)

A public, hosted API at `phenobrain.cs.tsinghua.edu.cn` (Tsinghua University), no API key required. Used in two different places, hitting different endpoints but sharing the same asynchronous submit-then-poll pattern (submit → receive task ID → poll until state is `SUCCESS`):

| Caller | Endpoint | Purpose |
|---|---|---|
| HPO extraction | `/extract-hpo` | Submit clinical text, receive a task ID |
| HPO extraction | `/query-extract-hpo-result` | Poll for extracted HPO terms |
| Validation tool | `/predict` | Submit HPO terms, receive a task ID |
| Validation tool | `/query-predict-result` | Poll for ranked disease predictions (returned as PhenoBrain RD codes) |
| Validation tool | `/disease-list-detail` | Translate RD codes into OMIM/ORPHA identifiers for ground-truth comparison |

### Dx29

Two validation-tool wrappers — `run_dx29_search.py` and `run_dx29_phrank.py` — both call the same configurable host, defaulting to `http://localhost:8080`, via different endpoints: `/api/v1/Search` and `/api/v1/Diagnosis/phrank`.
