# Transformer-Based Disease Retrieval Method

## Overview

The transformer-based disease retrieval method ranks rare diseases by comparing dense text embeddings of a patient profile against dense text embeddings of disease profiles.

Unlike semantic HPO methods, this method does not use the HPO ontology graph, ancestor propagation, or information-content scores to calculate similarity. Instead, it converts patient and disease information into text, embeds both sides with transformer encoder models, and ranks diseases by cosine similarity between the resulting vectors.

The method is implemented across five files:

```text
config.py        Model list, constants, cache paths, and pipeline settings
methods.py       Text construction and embedding backend utilities
retriever.py     Disease ranking, caching, canonical deduplication, and result construction
explanation.py   Structured explanations for transformer results
pipeline.py      Pipeline entry point used by the RareSim framework
```

## High-Level Method Logic

The transformer pipeline follows this sequence:

```text
Patient HPO terms + optional raw clinical text
        ↓
Build patient text
        ↓
Embed patient text with selected transformer model
        ↓
Disease label + disease HPO labels + optional disease description
        ↓
Build disease texts
        ↓
Embed disease texts with the same transformer model
        ↓
Compute cosine similarity between patient and disease embeddings
        ↓
Collapse aliases into canonical disease IDs
        ↓
Return ranked disease results with explanations
```

The final score is a dense embedding similarity score. Higher scores indicate that the patient profile text and disease profile text are more similar in the embedding space of the selected transformer model.

## Input Representation

### Patient Text

The patient text is built from direct patient HPO terms and optional raw clinical text.

Example:

```text
Patient phenotypes: Seizure; Global developmental delay; Hypotonia Patient description: The patient presents with...
```

The pipeline uses direct/raw HPO terms:

```python
patient.get_terms(use_propagated=False)
```

This means propagated ancestor terms are not used as transformer input.

Phenotype labels are placed before raw text so that, if transformer tokenization truncates the input, the primary phenotype signal is less likely to be removed.

### Disease Text

Each disease profile is converted into one text string containing:

1. Disease label
2. Direct disease HPO phenotype labels
3. Merged disease description, if available

Example:

```text
Disease: Rett syndrome Phenotypes: Seizure; Microcephaly; Loss of speech Description: Rett syndrome is characterized by...
```

The disease text is the representation that is embedded and compared against the patient embedding.

## Supported Transformer Models

The method supports multiple encoder-based transformer models:

```text
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
emilyalsentzer/Bio_ClinicalBERT
sentence-transformers/all-MiniLM-L6-v2
cambridgeltl/SapBERT-from-PubMedBERT-fulltext
dmis-lab/biobert-v1.1
```

These models are treated as separate retrieval variants during evaluation.

MiniLM is loaded through the `SentenceTransformer` library. The other models are loaded through HuggingFace `AutoTokenizer` and `AutoModel`.

## Embedding Backends

The pipeline supports two embedding backends.

### HuggingFace Encoder Backend

Used for models such as PubMedBERT, ClinicalBERT, SapBERT, and BioBERT.

The process is:

```text
Text
 ↓
Tokenizer
 ↓
AutoModel encoder
 ↓
Token embeddings
 ↓
Mean pooling over non-padding tokens
 ↓
L2 normalization
 ↓
Final text embedding
```

Because HuggingFace encoder models return token-level embeddings, the pipeline applies mean pooling to obtain one vector per input text.

### SentenceTransformer Backend

Used for MiniLM.

The SentenceTransformer backend directly returns sentence-level embeddings. The pipeline enables normalized embeddings so that cosine similarity can be computed with a dot product.

## Similarity Computation

Both patient and disease embeddings are L2-normalized. Therefore, cosine similarity can be computed as a dot product:

```python
scores = disease_embeddings @ patient_embedding
```

Each score compares one disease profile against the patient profile.

A higher score means the disease text is closer to the patient text in the embedding space of the selected model.

## Disease Embedding Cache

Disease embeddings are expensive to compute because every model must embed all disease profiles. To avoid recomputation, the pipeline stores persistent disease embedding caches on disk.

For each model, the cache contains:

```text
disease_ids.json
disease_labels.json
disease_texts.json
disease_embeddings.npy
metadata.json
```

The cache is model-specific. A separate cache is created for each transformer model.

The metadata file stores information such as:

```text
model name
model type
maximum sequence length
number of diseases
disease fingerprint
```

The disease fingerprint is a stable hash of disease IDs and disease texts. It is used to detect stale caches. If the disease profiles or text construction logic changes, the fingerprint changes and the cache is rebuilt.

## Patient Embedding Cache

Patient embeddings are cached in memory during a run.

The cache key is based on:

```text
model name + hash of patient text
```

This prevents repeated embedding of the same patient text with the same model during one execution.

Unlike disease embeddings, patient embeddings are not written to disk.

## Canonical Disease Deduplication

The disease database can contain multiple identifiers or aliases for the same underlying disease, such as OMIM and ORPHA identifiers.

The transformer model first scores disease profiles at the alias level. Then the retriever collapses alias-level results into canonical disease-level results.

This avoids cases where the same disease appears multiple times in the top-k ranking under different identifiers.

The deduplication process does the following:

1. Maps every disease ID to its canonical disease ID.
2. Keeps the highest-scoring alias as the representative result.
3. Preserves matched aliases for inspection.
4. Returns one ranked result per canonical disease.

This makes evaluation fairer because duplicate aliases do not waste top-k positions.

## Result Construction

Each final result is returned as a `SimilarityResult` object containing:

```text
canonical disease ID
disease label
similarity score
method/model name
category metadata
matched aliases
rank
structured explanation
```

The method name used in the result is the selected transformer model name. Therefore, evaluation can compare different transformer models separately.

## Explanations

The explanation system describes how each transformer score was produced.

The explanation records that:

```text
the method uses dense embeddings
the score is cosine similarity
the embeddings are L2-normalized
the method does not use IC values
the method does not use direct HPO overlap for scoring
```

Shared HPO labels may be shown in the explanation, but they are only descriptive. They help the user inspect why a result may look clinically plausible, but they do not drive the transformer score.

## Pipeline Entry Point

The `pipeline.py` file connects the transformer retriever to the general RareSim pipeline framework.

The main `run()` function:

1. Creates a `DiseaseRetriever` from the shared `AppContext`.
2. Prepares or loads disease embedding caches.
3. Runs each selected transformer model.
4. Calls the retriever to rank diseases.
5. Measures runtime.
6. Builds run statistics.
7. Sorts and formats results using the shared pipeline utilities.
8. Unloads the model backend after use.

The `run_default_model()` function runs only the default transformer model, which is MiniLM. This is useful for frontend or API usage where loading all transformer models would be inefficient.

The `main()` function allows the transformer pipeline to be run from the command line through the shared RareSim pipeline runner.

## Run Statistics

The transformer pipeline records run statistics such as:

```text
number of raw patient terms
number of propagated patient terms used
number of patient labels used
number of diseases scored
number of diseases skipped
runtime
```

For this method, propagated patient terms are recorded as zero because the transformer pipeline uses direct HPO terms only.

## Summary

The transformer-based disease retrieval method is a direct text-embedding retrieval approach. It builds text representations for patients and diseases, embeds them with encoder-based transformer models, and ranks diseases using cosine similarity. Disease embeddings are cached on disk per model, patient embeddings are cached in memory, and alias-level disease results are collapsed into canonical disease-level rankings.
