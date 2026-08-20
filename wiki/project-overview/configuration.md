# Configuration

RareSim's configuration is spread across a few deliberate layers: one environment variable that anchors every path, one file of build-time constants, one per-run config object that every similarity method reads, and per-method `config.py` files for method-specific tuning. This page covers each layer and where it lives.


## `RARESIM_ROOT` — the one required environment variable

`packages/raresim-core/src/raresim/utils/paths.py` is the single source of truth for every path in the project (ontology sources, artifacts, datasets, model caches, outputs). It reads `RARESIM_ROOT` from the environment after calling `load_dotenv()`, and **importing `paths.py` raises `KeyError` immediately if `RARESIM_ROOT` isn't set** — this is the most common reason a script fails before any real logic runs.

Because `load_dotenv()` is used, a `.env` file at the project root containing

```text
RARESIM_ROOT=/absolute/path/to/RareSim-Master-Project-
```

is the expected way to set it for local development, rather than exporting it in every shell session. See [Installation](../getting-started/installation.md) for the full setup sequence.

From `RARESIM_ROOT`, `paths.py` derives everything else:

```text
DATA_DIR, ONTOLOGY_DIR, DATASET_DIR, MODELS_DIR   source data, ontologies, benchmark datasets, model files
OUTPUTS_DIR, ARTIFACTS_DIR, SIMILARITY_DIR         built artifacts and method outputs (ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts")
ONTOLOGY_PATHS                                     dict of raw ontology source file paths (HPO, ORDO, MONDO, HOOM, HPOA, Orphadata, Monarch)
```

plus the paths to every built artifact file (`canonical_disease_profiles.json`, `hpo_labels.json`, `alias_to_canonical.json`, `information_content.json`, ancestor/parent files, metadata indices — the full list is in [Output](output.md)).


## Build-time behavior constants — `core/config.py`

Where `paths.py` says *where things live*, `core/config.py` says *how the build phase behaves*:

```text
APPLY_TRUE_PATH_RULE = True
    Propagate HPO terms upward through the ontology hierarchy when building
    disease profiles — a disease with a specific phenotype also implies its
    ancestor phenotypes. This is the standard true-path rule for HPO-based
    similarity, and it's why "propagated" term sets exist at all.

MIN_DISEASE_HPO_TERMS = 1
    Documents the intended minimum HPO term count for a disease profile to be
    kept.

CANONICAL_DISEASE_NAMESPACE = "ORPHA"
    Records the project decision that canonical disease profiles prefer
    Orphanet/ORPHA IDs wherever a reliable cross-source mapping exists.

EXAMPLE_PATIENT
    A small hard-coded patient (raw text + three HPO terms) used to seed
    outputs/artifacts/example_patient.json during the build, and as the
    default patient for method_runner.py-based script runs when no patient
    is supplied.
```


## Per-run configuration — `PipelineConfig`

`PipelineConfig` (defined in `types/result.py`) is the object every similarity method pipeline actually receives at run time. It's saved alongside every result, so each output records exactly how it was produced.

```text
top_k                    how many ranked results to keep (sort_and_rank() truncates to this)
use_propagated_terms     whether PatientProfile.get_terms() returns propagated_hpo_terms (default) or raw hpo_terms
ic_threshold              information-content cutoff used by semantic methods to filter overly generic HPO terms
use_canonical_profiles    whether AppContext loads canonical_disease_profiles.json (ORPHA-preferred) or the
                           alias-expanded disease_profiles.json
```

`PipelineConfig.terms_key` is a small but important accessor: it returns `"propagated_hpo_terms"` or `"hpo_terms"` depending on `use_propagated_terms`, so pipelines pick the right field consistently instead of each hardcoding one or the other.

**Both `PatientProfile.get_terms()` and `PipelineConfig` default toward propagated terms.**

`use_canonical_profiles` matters most for evaluation: benchmark ground truth sometimes uses OMIM or MONDO IDs directly, so scoring against the alias-expanded profile set (`use_canonical_profiles=False`) can be necessary for those IDs to match at all. See [dataset-format.md](../evaluation/dataset-format.md) and [evaluator-and-metrics.md](../evaluation/evaluator-and-metrics.md) for how alias matching works on the evaluation side.

Every batch runner under `scripts/evaluation/` builds its own `PipelineConfig` from CLI flags — see [batch-runners-and-shared-utilities.md](../evaluation/batch-runners-and-shared-utilities.md) for the exact defaults each runner uses.


## HPO extraction configuration — `hpo_extraction/_config.py`

Extraction-specific settings live separately from `core/config.py`, since they only apply when building a `PatientProfile` from raw text rather than pre-coded HPO terms:

```text
NEGATION_WORDS          phrases like "no", "not", "without", "denied", "negative for", "absence of"
NEGATION_WINDOW_SIZE    how far before a matched phrase to look for a negation cue (currently 50 characters)
HPO_BLOCKLIST           overly generic/structural HPO terms to always drop (e.g. HP:0000001, HP:0000118,
                         and terms representing inheritance, onset, clinical modifier, frequency, family history)
EXTRACTION_METHODS      the list of available backend names: dictionary, biomedical_ner, fast_hpo_cr,
                         chatgpt, phenobrain_api
BIOMEDICAL_NER_MODEL / BIOMEDICAL_NER_MIN_CONFIDENCE
                         HuggingFace NER model name and confidence cutoff for the biomedical_ner backend
```

`registry.py` also exposes an `EXTRACTION_METHODS` list — keep the two in sync if you add or remove a backend; nothing currently enforces they match automatically.


## Per-method configuration — `similarity_methods/<method>/config.py`

Every similarity method family has its own `config.py` for method-specific settings (model names/lists, thresholds, candidate pool sizes, and similar). These aren't centralized because they're genuinely method-specific — for example:

```text
similarity_methods/transformer/config.py   MODEL_LIST, CANDIDATE_POOL_SIZE
similarity_methods/llm/config.py            LLM_MODEL_LIST, MAX_NEW_TOKENS_RETRIEVAL
similarity_methods/hpo2vec/                 METHOD_NAMES, PIPELINE_NAME, MODEL_CACHE_DIR
```

If you're adding a new method, its `config.py` is where method-specific constants belong — see [adding-method.md](../evaluation/adding-method.md) for the evaluation-runner side of adding a method, and see [adding-new-method.md](../similarity-methods/adding-new-method.md) for adding a new similarity method in detail.


## Benchmark visualization configuration — `config.py`

Separately from all of the above, the benchmark visualization toolkit (`scripts/evaluation/benchmark_visualization/config.py`) has its own configuration surface — display labels, the dataset allow-list, and colors. This is documented in full in [visualizing-results.md](../evaluation/visualizing-results.md#adding-a-method-tool-or-dataset), since it's specific to that reporting layer rather than the core runtime.
