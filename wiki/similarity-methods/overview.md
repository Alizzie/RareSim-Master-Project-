# Similarity Methods Overview

RareSim ranks candidate rare diseases against a patient profile using seven independent method families, each living in its own `similarity_methods/<method>/` directory under `packages/raresim-core/src/raresim/`. They differ entirely in how they compute a score — set overlap, ontology-informed semantic similarity, TF-IDF weighting, learned embeddings, or a generative LLM — but every one of them consumes the same `PatientProfile`/`DiseaseProfile` inputs and returns the same `SimilarityResult`/`MethodResults` output shape (see [Output](../project-overview/output.md#the-standard-method-output-object--methodresults)), so they can be run, cached, evaluated, and compared interchangeably. This page is the map across all seven; each method's own page covers its formulas and internals in depth.

## The methods at a glance

| Family | Page | Score is based on | Typical range |
|---|---|---|---|
| Set-based | [set-based-methods.md](./set-based-methods.md) | Binary HPO term-set overlap (Jaccard, Dice, Overlap coefficient, Cosine) | [0, 1] |
| Semantic (BMA) | [semantic-methods.md](./semantic-methods.md) | Information content of the most informative common ancestor between term pairs, aggregated with Best Match Average | Resnik: [0, 1]  · Lin: [0, 1] · Jiang-Conrath: (0, 1] |
| TF-IDF | [tfidf-methods.md](./tfidf-methods.md) | IDF-weighted cosine similarity, over HPO terms or free text depending on mode | [0, 1] |
| HPO2Vec+ | [hpo2vec.md](./hpo2vec.md) | Cosine similarity of IC-weighted-averaged Word2Vec embeddings, trained on IC-weighted random walks over the HPO graph + disease associations | [-1, 1] in theory, [0, 1] in practice |
| Denoising Autoencoder | [denoising-autoencoder.md](./denoising-autoencoder.md) | Euclidean distance between autoencoder latent vectors of binary HPO vectors, converted to a similarity | (0, 1] |
| Transformer | [embedding.md](./embedding.md) | Cosine similarity of dense text embeddings from a biomedical sentence-transformer model | dot product of L2-normalized vectors |
| LLM | [llm.md](./llm.md) | A generative model's confidence label for a disease it proposed directly, mapped to a fixed numeric score | discrete: {0.1, 0.3, 0.5, 0.6, 0.9} |


## Standard file layout

Five of the seven families follow the same four-file layout:

```text
similarity_methods/<method>/
    config.py       constants, thresholds, paths, method name(s)
    methods.py       the scoring/embedding functions themselves
    pipeline.py       entry point: run(), main(), connects to the shared framework
    explanation.py     builds the method's "why this disease ranked here" block
```

That's set-based, semantic, TF-IDF, HPO2Vec+, and the denoising autoencoder. Transformer and LLM add a fifth file:

```text
similarity_methods/transformer/   config.py, methods.py, explanation.py, pipeline.py, retriever.py
similarity_methods/llm/           config.py, methods.py, explanation.py, pipeline.py, retriever.py
```

`retriever.py` holds a higher-level orchestration class (`DiseaseRetriever`, `LlmDiseaseRetriever`) that both of these methods need because they do more than score one patient-disease pair at a time — transformer manages a disk-backed embedding cache across models, and LLM manages prompt construction, generation, parsing, and validation against the local disease database.

Every family's `pipeline.py` connects to the shared runtime described in [Overview → Shared runtime infrastructure](../project-overview/overview.md#shared-runtime-infrastructure-core): it reads artifacts once via `AppContext`, builds results into the standard `SimilarityResult`/`MethodResults` shape, and is runnable standalone via `core/method_runner.run_similarity_method()` or through the batch evaluation harness (see [workflow-overview.md](../evaluation/workflow-overview.md)).


## Propagated vs. raw HPO terms

Most methods score against **propagated** HPO terms (the patient's/disease's terms plus all HPO ancestors — see [Overview → data flow](../project-overview/overview.md#end-to-end-data-flow)), which is the default from `PipelineConfig.terms_key`. Two methods deliberately opt out:

- **Transformer** uses `patient.get_terms(use_propagated=False)` — direct terms only. Phenotype labels are placed before raw clinical text in the constructed prompt so tokenizer truncation is less likely to cut the primary signal.
- **LLM** also uses direct terms only, for the same reason: propagating "Abnormality of the nervous system" alongside "Seizure" would just add generic ontology noise to a text prompt.

HPO2Vec+'s graph construction similarly uses **raw** (not propagated) disease-to-phenotype edges, on the reasoning that the IS-A edges already encode the ontology hierarchy, so adding propagated terms as direct disease-phenotype edges would double up and bias walks toward ancestor nodes.


## Every method converts to "higher is better"

`core/pipeline.sort_and_rank()` assumes higher score means better match. Methods whose natural output is a distance or loss convert before returning results:

```text
Jiang-Conrath:          similarity = 1 / (1 + distance)
Denoising autoencoder:   similarity = 1 / (1 + ||patient_latent − disease_latent||₂)
```


## Where to go next

- New to the project? Start with [Project Overview](../project-overview/overview.md) for how these methods fit into the wider pipeline.
- Adding a new method family to `raresim-core`? See [Adding a New Method](./adding-new-method.md).
- Wiring an existing method into the batch benchmark harness? See [adding-method.md](../evaluation/adding-method.md) — that page covers the evaluation-runner side, not the method implementation itself.
- Each method's own page: [Set-Based](./set-based-methods.md), [Semantic](./semantic-methods.md), [TF-IDF](./tfidf-methods.md), [HPO2Vec+](./hpo2vec.md), [Denoising Autoencoder](./denoising-autoencoder.md), [Transformer](./embedding.md), [LLM](./llm.md).
