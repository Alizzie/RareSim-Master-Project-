# Quick Start

This assumes you've completed [Installation](installation.md) — `RARESIM_ROOT` is set and `outputs/artifacts/` is populated. A few paths from here depending on what you're trying to do: run the full benchmark evaluation workflow, run one method's pipeline directly against a single patient, or use the web interface. The first is more thoroughly verified against actual source code, so start there if you're unsure.


## Option A: run the benchmark evaluation workflow (recommended first run)

This is the most reliably documented path in this wiki — every step below is grounded in the actual runner scripts, not a summary of them. It also exercises far more of the system than a single method call: artifact loading, patient construction, method scoring, caching, and metric computation.

### Before running this: get a benchmark dataset

`data/datasets/` is gitignored (large files) — it isn't populated by cloning the repo or by the artifact-build steps in [Installation](installation.md), which only handle ontology sources (HP, ORDO, MONDO, HOOM, …), not benchmark test sets. You need the dataset file itself (e.g. `MME.json`) in place under `data/datasets/phenobrain_testdata/` before the command below will find anything to run.

- [dataset-available.md](../evaluation/dataset-available.md) documents where each benchmark dataset actually comes from (PhenoBrain benchmark → Zenodo, Phenopacket Store → GitHub, GA4GH Phenopackets → Zenodo, MyGene2 → Harvard Dataverse), if you need the original source rather than a pre-placed file.
- Phenopacket-format and MyGene2 sources need a standardization pass before they match the test-set JSON shape the batch runners expect — see `standardize_phenopackets.py` / `standardize_mygene2.py` under `scripts/evaluation/data_prep/`, and [dataset-format.md](../evaluation/dataset-format.md) for the target schema.

```bash
# run one method family against a small benchmark test set
python scripts/evaluation/run_set_based.py \
    --test-set data/datasets/phenobrain_testdata/MME.json \
    --limit 5

# score it
python scripts/evaluation/evaluator.py --dataset MME
```

Check `outputs/evaluation/MME/MME_evaluation_summary.txt` for a human-readable result. If that ran cleanly, you have a working install. From here:

- Add more method runners (`run_semantic`, `run_tfidf`, `run_hpo2vec`, …) — see [batch-runners-and-shared-utilities.md](../evaluation/batch-runners-and-shared-utilities.md).
- Drop `--limit 5` once you're ready to run the full dataset.
- See [workflow-overview.md](../evaluation/workflow-overview.md) for the complete recommended command sequence, and [dataset-available.md](../evaluation/dataset-available.md) for what other benchmark datasets exist.
- Once you have results for a few methods, [visualizing-results.md](../evaluation/visualizing-results.md) turns them into comparison figures and an HTML report.


## Option B: run one similarity method's pipeline directly

Each method's `pipeline.py` is directly runnable as a script — this exercises `raresim-core` on its own, without going through the batch-evaluation layer.

**Check GPU availability first** if you're running transformer, LLM, or autoencoder (all three use a GPU when available):

```bash
nvidia-smi
```

**Transformer** — builds an embedding cache for all configured models on first run (~10 min on GPU); reused on subsequent runs, so ranking after that is near-instant:

```bash
mkdir -p outputs/similarity_methods/transformer
CUDA_VISIBLE_DEVICES=4 nohup python packages/raresim-core/src/raresim/similarity_methods/transformer/pipeline.py \
    >| outputs/similarity_methods/transformer/transformer_log.txt 2>&1 &

# watch progress
tail -f outputs/similarity_methods/transformer/transformer_log.txt
# check whether it's still running
ps aux | grep transformer/pipeline.py
```

**LLM** — runs Mistral-7B-Instruct-v0.2 for disease retrieval + explanation, 4-bit quantized (fits ~6GB GPU memory); takes roughly 3–5 minutes per run for retrieval + explanation of 10 results:

```bash
CUDA_VISIBLE_DEVICES=4,5 nohup python packages/raresim-core/src/raresim/similarity_methods/llm/pipeline.py \
    >| outputs/similarity_methods/llm/llm_log.txt 2>&1 &

tail -f outputs/similarity_methods/llm/llm_log.txt
```

**Autoencoder:**

```bash
mkdir -p outputs/similarity_methods/autoencoder
nohup python packages/raresim-core/src/raresim/similarity_methods/autoencoder/pipeline.py \
    >| outputs/similarity_methods/autoencoder/autoencoder_log.txt 2>&1 &
```

Swap the GPU index(es) in `CUDA_VISIBLE_DEVICES` for whichever `nvidia-smi` shows as free. `nohup ... &` or `screen ... &` backgrounds the process and keeps it running after you disconnect — useful when running these on a remote server over SSH.

The CPU-only method families (`set_based`, `semantic`, `tfidf`, `hpo2vec`) follow the same file layout (`similarity_methods/<method>/pipeline.py`), so the same direct-invocation pattern is expected to apply — e.g. `python packages/raresim-core/src/raresim/similarity_methods/set_based/pipeline.py`.

Without an explicit patient, these default to the example patient from installation.


## Option C: the web interface

There's also a live GUI for entering a patient's HPO terms or clinical text interactively and seeing ranked results, rather than running a script — a FastAPI backend (`raresim-backend`) plus a Vue frontend (`raresim-frontend`).

```bash
# terminal 1 — backend
uvicorn raresim_api.main:app --reload --port 8000

# terminal 2 — frontend
cd packages/raresim-frontend
npm install
npm run dev
```

Then open the frontend's local dev URL in a browser. Reach for this if you want to try a patient interactively.


## Where output goes

All three options write into `outputs/` — see [Output](../project-overview/output.md) for the full map of what gets written where.
