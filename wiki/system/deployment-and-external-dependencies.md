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

### PhenoBrain (Hosted API and Local)

PhenoBrain is used in two different ways depending on the caller

#### Hosted API

A public, hosted API at `phenobrain.cs.tsinghua.edu.cn` (Tsinghua University), no API key required. **HPO extraction** (`phenobrain.py`) always uses this path. The **validation-tool comparison** use this same hosted path via `run_phenobrain.py`, hitting different endpoints but sharing the same asynchronous submit-then-poll pattern (submit → receive task ID → poll until state is `SUCCESS`):

| Caller | Endpoint | Purpose |
|---|---|---|
| HPO extraction | `/extract-hpo` | Submit clinical text, receive a task ID |
| HPO extraction | `/query-extract-hpo-result` | Poll for extracted HPO terms |
| Validation tool | `/predict` | Submit HPO terms, receive a task ID |
| Validation tool | `/query-predict-result` | Poll for ranked disease predictions (returned as PhenoBrain RD codes) |
| Validation tool | `/disease-list-detail` | Translate RD codes into OMIM/ORPHA identifiers for ground-truth comparison |


#### Local (validation-tool comparison only)

The validation-tool comparison also runs PhenoBrain **locally**.

**Deployment.** This is a self-hosted deployment built from the authors' own pipeline ([`xiaohaomao/timgroup_disease_diagnosis`](https://github.com/xiaohaomao/timgroup_disease_diagnosis) on GitHub), run against RareSim's own datasets via that repo's evaluation entry point, `core/script/test/test_optimal_model.py`, on the `INTEGRATE_CCRD_OMIM_ORPHA` knowledge base.

**Two-step process.**

```text
1. Run PhenoBrain locally     upstream repo's test_optimal_model.py,
                              against INTEGRATE_CCRD_OMIM_ORPHA —
                              produces a raw per-model rank export (.xlsx)

2. Standardize the export      run_phenobrain_local.py — does NOT run
                              PhenoBrain itself, only converts step 1's
                              output into the same summary TSV format
                              compare_methods.py expects from every
                              other validation tool
```

**Raw export shape.** Five fixed columns (`DATA_RANK`, `DISEASE_CODE`, `DISEASE_NAME`, `HPO_CODE`, `HPO_NAME`); everything else is a per-model rank column, one per internally configured scoring backbone / random-seed variant. The confirmed full set of 16 model columns:

```text
BOQAModel-dp1.0-Random    CNB-Random                 CosineModel-Random
GDDPFisherModel-MinIC-Random   HPOProbMNB-Random      ICTODQAcross-Ave-Random
JaccardModel-Random         MICA-QD-Random             MICAJC-QD-Random
MICALin-QD-Random           MinIC-QD-Random            NN-Mixup-Random-1
RBPModel-Random              RDDModel-Ances-Random      SimGICModel-Random
SimTOModel-Random
```

`DISEASE_CODE` and `HPO_CODE` are stringified Python lists (e.g. `['RD:6786']`). The export is native `.xlsx`, read via `openpyxl` (first sheet); a `.csv`/`.tsv` export also works with delimiter auto-detection.

By default, every model column found is standardized in one pass:

```bash
python3 run_phenobrain_local.py --input phenobrain_raw/0.1.27.xlsx --dataset 0.1.27
```

Each model gets its **own output folder**:

```python
def out_dir_for_model(base_dir: Path, model: str) -> Path:
    return base_dir / f"phenobrain ({model})_benchmarks"
```

This is the mechanism behind the 16-variant breakdown visible in the evaluation chapter's method-comparison figures: because `compare_methods.py`'s auto-discovery treats each `phenobrain (<model>)_benchmarks/` folder as its own method (folder name minus `_benchmarks` → method name.

### Dx29

Two validation-tool wrappers — `run_dx29_search.py` and `run_dx29_phrank.py` — both call the same configurable host, defaulting to `http://localhost:8080` (unlike PhenoBrain's public URL, implying Dx29 is normally **self-hosted**, not called over the public internet), via different endpoints: `/api/v1/Search` and `/api/v1/Diagnosis/phrank`.
