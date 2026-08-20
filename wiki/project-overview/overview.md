# Project Overview

## What RareSim is

RareSim retrieves and ranks candidate rare diseases for a patient, given either structured HPO (Human Phenotype Ontology) phenotype terms or raw clinical text. It implements several independent similarity/retrieval method families (set-based, semantic, TF-IDF, HPO2Vec, a denoising autoencoder, transformer embedding retrieval, and direct LLM retrieval) against a shared, precomputed corpus of disease profiles, and provides shared infrastructure — patient/disease data models, a common output schema, evaluation tooling, and a benchmarking/visualization layer — so those methods can be compared on equal footing.


## Repository layout

Full top-level tree:

```text
RareSim-Master-Project-/
├── data/                     gitignored — datasets, ontologies, models, patient_profiles (see data/README.md)
│   ├── datasets/
│   ├── models/
│   ├── ontologies/
│   ├── patient_profiles/
│   └── README.md
├── docs/                     drawio diagrams, notebooks, useful_cmd.txt
│   ├── drawio/
│   ├── notebooks/
│   └── useful_cmd.txt
├── outputs/                  gitignored, generated at runtime — see Output
├── packages/
│   ├── raresim-backend/        API layer
│   ├── raresim-cli/            Terminal interface
│   ├── raresim-core/           Core retrieval framework
│   └── raresim-frontend/       User interface
├── scripts/
│   ├── evaluation/             Batch runners + evaluator + benchmark visualization (see the Evaluation wiki section)
│   ├── experiments/            Experiment scripts (e.g. raw vs. propagated HPO terms)
│   └── validation_tools/       Runners for external tools (LIRICAL, Phenomizer, PhenoBrain, Dx29) used as comparison baselines
├── third_party/               externally cloned tools, gitignored (fast_hpo_cr)
├── wiki/                      this documentation site
├── pyproject.toml             root: dev tooling + uv workspace (see Installation)
├── pyrightconfig.json
├── requirements.txt / requirements_server.txt
├── setup.sh                   bootstrap script (see Installation)
├── package.json / package-lock.json   root-level JS tooling (the wiki's build, not raresim-frontend's own)
└── README.md
```

`raresim-core` is the framework everything else is built on: it owns the data models, the ontology/artifact build pipeline, the similarity methods themselves, and the shared runtime infrastructure. The `scripts/` tree is a separate, script-only layer for batch evaluation, benchmarking, and comparison against external tools — it consumes `raresim-core` but isn't part of the installable package.

`data/` and `third_party/` are both gitignored, same as `outputs/` — none of the three come from cloning the repo. `data/ontologies/` and `data/models/` get populated by the [Installation](../getting-started/installation.md) bootstrap steps; `data/datasets/` (benchmark test sets) needs separate manual download — see the note in [Quick Start](../getting-started/quick-start.md#before-running-this-get-a-benchmark-dataset) and `data/README.md` itself.


## `raresim-core` directory tree

```text
packages/raresim-core/src/raresim/
|
|-- types/              PatientProfile, DiseaseProfile, result schemas
|-- utils/               Paths, IO, normalization, text utilities, math helpers
|-- build/               Offline artifact construction entry points
|-- ontology/            Ontology loading, phenotype merging, profile construction
|-- hpo_extraction/      Extraction of HPO terms from patient text
|-- core/                Runtime context, pipeline helpers, explanations, cache
|-- similarity_methods/  Internal RareSim similarity and retrieval methods
`-- analysis/            Method comparison utilities
```


## The two core data objects

Everything in RareSim revolves around two objects, defined in `types/schemas.py`.

**`PatientProfile`** — the query side:

```text
patient_id
raw_text
hpo_terms              positive/observed phenotype terms
propagated_hpo_terms   hpo_terms plus all HPO ancestors
excluded_hpo_terms     phenotypes explicitly noted absent
```

**`DiseaseProfile`** — what the system scores against:

```text
disease_id
label
profile_type
hpo_terms
propagated_hpo_terms
negative_hpo_terms
ordo_label / ordo_description
mondo_label / mondo_description
hoom_label / hoom_description
merged_description
source_ids
aliases
category_source_id
canonicalized_to_orpha
term_provenance
```

Positive and excluded patient terms are kept strictly separate: `hpo_terms`/`propagated_hpo_terms` are positive evidence, `excluded_hpo_terms` is negative evidence, and a similarity method only acts on excluded terms if it explicitly implements that logic (see [`run_set_jaccard_penalized.py`](../evaluation/batch-runners-and-shared-utilities.md#run_set_jaccard_penalizedpy) for the one method that currently does).


## End-to-end data flow

```text
1. Ontology + build phase (offline, run once / when ontologies update)
   raw OWL/HPOA/XML/TSV sources
     -> ontology/loaders.py            parse into common record shapes
     -> ontology/phenotype_merge.py    dedupe annotations, split out negative assertions
     -> ontology/disease_profiles.py   build canonical DiseaseProfile objects (ORPHA-preferred)
     -> ontology/ic.py                 compute HPO information content
     -> ontology/disease_ancestors.py  build ORDO category chains
     -> build/build_shared_artifacts.py  writes outputs/artifacts/*.json

2. Patient input
   raw clinical text --(hpo_extraction)--> hpo_terms
   or hpo_terms supplied directly
     -> PatientProfile (hpo_terms, propagated_hpo_terms, excluded_hpo_terms)

3. Runtime scoring
   AppContext.load(patient)   loads outputs/artifacts/*.json once, shared by every method
     -> a similarity method scores PatientProfile against every DiseaseProfile
     -> SimilarityResult objects (one per ranked disease)
     -> core/pipeline.sort_and_rank()   ranks, truncates to top_k
     -> MethodResults                    the standard output object for one method run

4. Consumption
   MethodResults -> saved JSON / run cache / API response / frontend / evaluation cache
```


## HPO extraction (raw text → HPO terms)

When a patient arrives as raw clinical text rather than pre-coded HPO terms, `hpo_extraction/` bridges the gap:

```text
raw clinical text -> extractor(s) -> ExtractionResult list -> deduplicate
    -> hpo_terms -> propagate through HPO ancestors -> propagated_hpo_terms
    -> PatientProfile
```

Five extraction backends are available (`registry.py` / `_config.py` → `EXTRACTION_METHODS`): `dictionary` (fast exact HPO-label matching), `biomedical_ner` (HuggingFace biomedical NER), `fast_hpo_cr` (a third-party lexical/morphological concept recognizer), `chatgpt` (LLM-based phrase extraction with local HPO mapping — the model identifies phenotype phrases but never invents HPO IDs itself), and `phenobrain_api` (an external async API). `ensemble.py` orchestrates whichever backends are selected, deduplicates their output, and builds the final patient profile. Negation handling (`NEGATION_WORDS`, a configurable lookback window) and a blocklist of overly generic HPO terms (`HPO_BLOCKLIST`) are applied uniformly across backends — see [Configuration](configuration.md).

This package prepares patient-side input; it is not itself a disease-similarity method.


## Similarity methods

Standard per-method file layout:

```text
similarity_methods/<method>/
    config.py
    methods.py
    explanation.py
    pipeline.py
```

Transformer and LLM methods add a `retriever.py` for higher-level orchestration:

```text
similarity_methods/transformer/   config.py, methods.py, explanation.py, pipeline.py, retriever.py
similarity_methods/llm/           config.py, methods.py, explanation.py, pipeline.py, retriever.py
```

Implemented method families:

```text
similarity_methods/
    set_based/      set_cosine, set_jaccard, set_dice, set_overlap, set_jaccard_penalized
    semantic/        HPO-structure + information-content methods (Resnik/Lin/Jiang-Conrath style, BMA)
    tfidf/           TF-IDF over HPO terms or raw text
    hpo2vec/         HPO term embeddings
    autoencoder/     denoising autoencoder over disease HPO profiles
    transformer/     biomedical sentence-transformer embedding retrieval
    llm/             direct LLM disease retrieval
    registry.py
```

Every method ultimately returns a list of `SimilarityResult` objects (or plain dicts with the same fields), which `core/pipeline.sort_and_rank()` sorts by score, ranks, and truncates to `top_k`. Because ranking assumes **higher score is better**, any method that naturally produces a distance/loss value has to convert it to a similarity score before results reach `sort_and_rank()`.

For batch evaluation of these methods against benchmark datasets, see [workflow-overview.md](../evaluation/workflow-overview.md) and the rest of the Evaluation section.


## Shared runtime infrastructure (`core/`)

`core/` is not a similarity algorithm — it's the shared plumbing every method relies on so results look consistent regardless of which method produced them.

```text
core/config.py       build-time behavior constants + the example patient seed (see Configuration)
core/context.py      AppContext — loads outputs/artifacts/*.json once, shared by every method
core/pipeline.py      build_run_stats(), sort_and_rank() — common post-scoring finalization
core/method_runner.py  run_similarity_method() — boilerplate for running one pipeline as a script
core/cache.py         save_run_cache() / load_run_cache() — multi-method run caching (see Output)
core/explanation/      shared schema + builders so every method's "why this disease ranked here"
                       explanation has the same shape (HPO-based or token-based)
analysis/method_comparison.py
                       builds RRF consensus + Jaccard agreement across methods for one patient case
                       (this is a live, single-case comparison — not the offline benchmark evaluator)
```

`AppContext` is central: it loads disease profiles, HPO labels, information content values, HPO/disease ancestors, disease metadata, and alias mappings exactly once, and every method reads from that same shared object rather than opening artifact files itself. Whether it loads canonical (ORPHA-preferred) or alias-expanded disease profiles is controlled by `PipelineConfig.use_canonical_profiles` — see [Configuration](configuration.md).

`analysis/method_comparison.py` is worth distinguishing from the evaluation layer: it answers "which methods agree on this one patient, right now" (consensus via reciprocal rank fusion, pairwise Jaccard agreement between method candidate sets) without any ground truth — agreement is not the same thing as correctness. The benchmark evaluator in `scripts/evaluation/evaluator.py` is the tool that actually scores methods against known-correct diagnoses; see [evaluator-and-metrics.md](../evaluation/evaluator-and-metrics.md).


## Where to go next

- New to the project and setting it up? Start with [Installation](../getting-started/installation.md) and [Quick Start](../getting-started/quick-start.md).
- Configuring build behavior, pipeline runs, or HPO extraction? See [Configuration](configuration.md).
- Wondering what files a run actually produces and where? See [Output](output.md).
- Running or extending the benchmark evaluation? Start at [Evaluation → Workflow Overview](../evaluation/workflow-overview.md).
