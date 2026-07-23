# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Alizzie/RareSim-Master-Project-.git
cd RareSim-Master-Project-
```

If you already have the repo cloned:

```bash
git pull origin main
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

## 3. Install dependencies and the project itself

**Recommended: `uv`.** From the project root, this resolves and installs the whole workspace in one step:

```bash
uv sync
```

**Without `uv`, for general/CPU-only development:**

```bash
pip install -e packages/raresim-core
```

**If you'll run the GPU-backed pipelines (transformer, LLM, autoencoder)** on the project's GPU server — install the pinned, CUDA-compatible dependency set instead of relying on `uv sync`'s normal resolution, since `pyproject.toml`'s unpinned/general dependency versions are newer than what the server's CUDA driver supports:

```bash
pip install -r requirements_server.txt
pip install -e . --no-deps
```

`requirements_server.txt` pins versions specifically for CUDA 12.4 (as used on the project's GPU server, `anna.ifi.uzh.ch`) — notably `torch==2.4.0+cu124` via `--extra-index-url https://download.pytorch.org/whl/cu124`, plus `transformers==4.44.0`, `numpy==1.26.4`, `bitsandbytes==0.49.2`, `accelerate==1.13.0`, and `sentencepiece`. Each pin exists for a specific reason:

```text
torch 2.4.0+cu124     required for CUDA 12.4 on the anna server (newer torch needs a driver update)
transformers 4.44.0   newer versions require torch>=2.6
numpy 1.26.4          must stay <2 — conflicts with system numexpr/bottleneck otherwise
bitsandbytes 0.49.2   required for 4-bit quantization of LLM models
accelerate 1.13.0     required for device_map="auto" in model loading
sentencepiece         required for Mistral/LLaMA tokenizers
```

If you're on a machine without that specific CUDA setup (e.g. local CPU-only development via `uv sync` or the plain `pip install -e packages/raresim-core` above), these exact pins don't apply to you — but if you hit version-mismatch errors running the transformer/LLM/autoencoder pipelines anywhere, this list is the first thing to check.

`pip install -e . --no-deps` (root install, dependency resolution skipped) is specifically the server-path form — it assumes `requirements_server.txt` already pinned everything precisely, so it shouldn't be used as a substitute for `uv sync` or the plain per-package `pip install -e packages/raresim-core` above.

## 4. Configure `RARESIM_ROOT`

`raresim.utils.paths` reads `RARESIM_ROOT` from the environment via `load_dotenv()`, and importing it raises `KeyError` immediately if the variable isn't set — every script that touches paths (nearly all of them) fails before doing anything else until this is set.

Create a `.env` file at the project root:

```text
RARESIM_ROOT=/absolute/path/to/RareSim-Master-Project-
```

Use an absolute path. See [Configuration](../project-overview/configuration.md#raresim_root--the-one-required-environment-variable) for what gets derived from this.

## 5. Bootstrap: third-party tools, ontology sources, and artifacts

The project ships a bootstrap script, `setup.sh`, that runs all three required steps in order:

```bash
./setup.sh
```

Which runs, in order:

```bash
# 1/3 — REQUIRED. fast_hpo_cr is a runtime dependency of the HPO
#       extraction methods, not an optional extra.
python -m raresim.build.setup_third_party

# 2/3 — downloads HPO, ORDO, MONDO, HOOM, HPOA, Orphadata Product 4,
#       and Monarch source files into data/ontologies/. Already-downloaded
#       files are skipped.
python -m raresim.build.load_ontologies_to_local

# 3/3 — parses everything above, merges annotations, canonicalizes disease
#       IDs, propagates HPO terms, computes information content, and writes
#       outputs/artifacts/*.json — see Output for the full artifact list.
python -m raresim.build.build_shared_artifacts
```

Building the artifacts can take a while the first time.

## 6. Verify the install

```bash
ls outputs/artifacts/
```

You should see `canonical_disease_profiles.json`, `hpo_labels.json`, `hpo_ancestors.json`, `information_content.json`, `example_patient.json`, and the rest of the list in [Output](../project-overview/output.md). If any are missing, re-check step 5 (did all three bootstrap steps succeed?) and step 4 (`RARESIM_ROOT` set correctly?).

Once artifacts exist, continue to [Quick Start](quick-start.md) to run something end-to-end.

---

## Reconnecting later (already set up)

```bash
cd /path/to/RareSim-Master-Project-
source venv/bin/activate
```

If dependencies changed since your last pull:

```bash
git pull origin main
pip install -r requirements_server.txt   # if using GPU pipelines
pip install -e . --no-deps
```
