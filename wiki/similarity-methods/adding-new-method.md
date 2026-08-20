# How to Add a New Similarity Method

This page covers adding a new method family to `raresim-core` itself — a new `similarity_methods/<method>/` package that scores patients against diseases and plugs into the shared pipeline. If you've already got a method and just want to run it against benchmark datasets, that's a separate step — see [adding-method.md](../evaluation/adding-method.md) in the Evaluation section instead.


## 1. Create the package directory

```text
packages/raresim-core/src/raresim/similarity_methods/<your_method>/
    config.py
    methods.py
    pipeline.py
    explanation.py
```

That four-file layout covers every existing method family except transformer and LLM, which add a `retriever.py` (a higher-level orchestration class) because they manage state beyond a single patient-disease comparison — a disk-backed embedding cache, or prompt construction and generation across a batch of candidates. Add `retriever.py` only if your method has that kind of cross-cutting state to manage; otherwise the four-file layout is the norm — see [Similarity Methods Overview → standard file layout](./overview.md#standard-file-layout).


## 2. `config.py` — constants your method needs

Every existing method keeps its own tunable constants here rather than in the shared `core/config.py` (see [Configuration → per-method configuration](../project-overview/configuration.md#per-method-configuration--similarity_methodsmethodconfigpy)), because they're genuinely method-specific: model names, thresholds, cache paths, candidate pool sizes. At minimum you'll want:

```text
PIPELINE_NAME          identifies your method in results, caches, and CLI output
<method-specific>       thresholds, model lists, cache directory, etc.
```

If your method has multiple named variants (like semantic's Resnik/Lin/Jiang-Conrath, or transformer's five model options), a method map here is the established pattern — see semantic's and transformer's `config.py` for the shape.


## 3. `methods.py` — the scoring logic

This is where the actual similarity computation lives. Two things every method needs to get right:

**Higher score = better match.** `core/pipeline.sort_and_rank()` assumes this. If your natural output is a distance or a loss, convert it before returning — the established pattern is `similarity = 1 / (1 + distance)`, used by both Jiang-Conrath (semantic) and the denoising autoencoder.

**Decide propagated vs. raw terms deliberately, and document the choice.** `PatientProfile.get_terms()` and `PipelineConfig` both default to propagated terms (terms plus all HPO ancestors). Most methods use that default. See [Overview → propagated vs. raw](./overview.md#propagated-vs-raw-hpo-terms) for how the existing methods split on this.


## 4. `explanation.py` — the "why this ranked here" block

Every method builds into the shared explanation schema from `core/explanation/` (see [Overview → shared runtime](../project-overview/overview.md#shared-runtime-infrastructure-core)), so the frontend/API can render any method's explanation uniformly.

**If your score does *not* come from direct HPO term overlap** (an embedding, a distance, a generated confidence — anything where "matched HPO terms" is not literally the formula computing the score), say so explicitly in the explanation, the way HPO2Vec+, the denoising autoencoder, transformer, and LLM all do:

```text
uses_ic_values
uses_dense_embeddings
uses_direct_hpo_overlap_for_score
score_note              plain-language statement of what actually drove the score
interpretation_note     clarifies that matched_terms, if shown, are descriptive only
```

See [Overview → the explanation object](./overview.md#the-explanation-object-shared-shape-method-specific-content) for the full pattern and which existing methods use it.


## 5. `pipeline.py` — the entry point

This is what connects your method to the shared framework and to the outside world (CLI, batch runners, API). The established shape, seen across every existing method:

```text
run(patient, selected, config, ctx)
    creates whatever retriever/scorer your method needs (from the shared AppContext, via ctx)
    scores the patient against diseases
    measures runtime
    builds run statistics (see below)
    sorts and ranks results via core/pipeline.sort_and_rank()
    returns MethodResults

main()
    lets the pipeline run standalone from the command line, through
    core/method_runner.run_similarity_method() — the shared convenience
    wrapper that supplies a default patient + default PipelineConfig()
    when none are given, and saves both combined and per-method result files
```

If your method loads a model or another expensive resource, unload/release it at the end of `run()` — transformer and LLM both do this explicitly (`unload_pipeline`) since these run on shared GPU resources.


## 6. Run statistics

Every method's pipeline records the same shape of run statistics via `core/pipeline.build_run_stats()`:

```text
number of raw patient terms
number of propagated patient terms used (or, for text-based methods, patient text tokens/labels used)
number of diseases scored
number of diseases skipped, and why (no terms in vocabulary, empty vector, etc.)
runtime
```

This is what feeds `RunStats` in the final `MethodResults` object (see [Output](../project-overview/output.md#the-standard-method-output-object--methodresults)) — it's what actually happened during the run, distinct from `PipelineConfig`, which is what was requested.


## 7. Register the method

New methods need to be discoverable by name for the shared framework and CLI to find them — `similarity_methods/registry.py` is where existing methods are listed (see [Project Overview → similarity methods](../project-overview/overview.md#similarity-methods)).


## 8. Wire it into batch evaluation (separate step)

Once your method works standalone, adding a `run_<method>.py` batch runner so it can be evaluated against benchmark datasets is a separate task with its own conventions — see [adding-method.md](../evaluation/adding-method.md) in the Evaluation section.
