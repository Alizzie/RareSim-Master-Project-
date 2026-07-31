# HPO2Vec+ Similarity Method

## Overview

HPO2Vec+ learns dense vector embeddings for every HPO term and disease node by training Word2Vec on **IC-weighted random walks** over a combined ontology and disease graph. Patient and disease profiles are each collapsed into a single embedding vector via IC-weighted averaging, and similarity is the cosine of the angle between those two vectors.

Unlike semantic methods, HPO2Vec+ does not compute pairwise term-to-term similarity using ancestor structure at inference time. Instead, the ontology structure and disease associations are baked into the embeddings during training. At inference, comparison is a single cosine operation between two dense vectors.

The method is implemented across five files:

```text
config.py        Walk parameters, embedding settings, model cache path
methods.py       Graph construction, IC-weighted walks, Word2Vec training, embedding
pipeline.py      Pipeline entry point, model loading/caching, scoring loop
explanation.py   Method-specific explanation block and base explainer delegation
```

---

## High-Level Method Logic

The HPO2Vec+ pipeline follows this sequence:

```text
HPO parents (IS-A edges) + disease profiles (HAS-PHENOTYPE edges)
        ↓
Build combined graph
        ↓
Generate IC-weighted random walks from every node
        ↓
Train Word2Vec (Skip-gram) on the walks
        ↓
For each patient:
    IC-weighted average of patient term embeddings
        ↓
    For each disease:
        IC-weighted average of disease term embeddings
            ↓
        Cosine similarity between patient and disease vectors
            ↓
    Rank diseases by score
```

---

## Step 1: Build the Graph

A single adjacency list is built from two edge types:

**IS-A edges** (from `hpo_parents.json`): connect each child HPO term to its parents, bidirectionally. Walks can traverse up and down the ontology hierarchy.

**HAS-PHENOTYPE edges** (from `disease_profiles.json`): connect each disease to its raw HPO terms, bidirectionally. Walks can hop from a disease node into its phenotypes and then up the hierarchy.

Raw HPO terms (not propagated) are used for HAS-PHENOTYPE edges because IS-A edges already capture the hierarchy. Adding propagated terms would create redundant edges that bias walks toward ancestor nodes.

```python
graph[child].append(parent)
graph[parent].append(child)
graph[disease_id].append(hpo_id)
graph[hpo_id].append(disease_id)
```

All nodes — both HPO terms and disease IDs — live in the same graph.

---

## Step 2: IC-Weighted Random Walks

Walks are biased using Information Content values and two structural parameters borrowed from Node2Vec.

### Transition Probabilities

For each step from `current` to a candidate `neighbour`:

```
weight = IC(neighbour) × bias
```

The bias depends on the walk history:

| Condition | Bias | Effect |
|-----------|------|--------|
| First step (no previous node) | 1.0 | Uniform — no history to bias from |
| `neighbour == previous` | `1 / p` | Penalizes returning to the previous node |
| `neighbour` is adjacent to `previous` | 1.0 | Neutral — distance-1 from previous |
| Otherwise | `1 / q` | Distance-2 from previous — DFS vs BFS control |

Disease nodes have no IC value and default to IC weight = 1.0.

Probabilities are normalized to sum to 1.0 after applying both IC and bias weights.

### Walk Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `p` | 1.0 | Return parameter. Lower p = more backtracking toward previous node |
| `q` | 0.5 | In-out parameter. q < 1 = DFS bias (explore deeper); q > 1 = BFS bias (stay local) |
| `WALK_LENGTH` | 80 | Number of steps per walk |
| `WALKS_PER_NODE` | 10 | Number of walks starting from each node |

Walks from all nodes are shuffled before being passed to Word2Vec so the model does not see all walks from the same node consecutively.

---

## Step 3: Train Word2Vec

The random walks are treated as sentences and fed to a **Skip-gram Word2Vec** model. Each HPO term ID and disease ID is a token. Terms that co-occur frequently within the same walk window receive similar embeddings.

| Parameter | Value |
|-----------|-------|
| `EMBEDDING_DIM` | 128 |
| `WINDOW_SIZE` | 10 |
| `MIN_COUNT` | 1 |
| `EPOCHS` | 5 |
| `WORKERS` | 4 |

---

## Step 4: Embed Term Sets

Each patient and disease HPO term set is collapsed into one vector via **IC-weighted averaging**:

```
embedding = Σ (IC(t) × vec(t))  /  Σ IC(t)   for t in terms ∩ vocab
```

- Terms not in the Word2Vec vocabulary are silently skipped.
- Terms with no IC value (e.g. disease nodes used as patient terms) default to weight 1.0.
- Returns `None` if no terms in the set have an embedding.

---

## Step 5: Cosine Similarity

```
score = cosine(patient_embedding, disease_embedding)
```

Score is in **[-1, 1]** but in practice stays in **[0, 1]** for non-negative IC-weighted vectors.

---

## Model Caching

Trained Word2Vec models are saved to `model_cache/` so they do not need to be retrained on every run. The cache is keyed on the training configuration. Delete the cache directory to force retraining.

---

## Important: Score vs HPO Overlap

The score is driven entirely by **embedding geometry**, not by direct HPO term overlap. The matched/unmatched term fields shown in the explanation are computed from raw HPO set intersection for readability only and do not affect the score. The explanation explicitly notes this.

---

## Explanation Fields

| Field | Description |
|-------|-------------|
| `summary` | Score, direct overlap count, and clarification that score is embedding-based |
| `coverage` | Patient/disease % matched, term counts (descriptive only) |
| `matched_terms` | HPO terms shared between patient and disease (descriptive only) |
| `unmatched_patient_terms` | Patient terms not in the disease (descriptive only) |
| `embedding_method` | `hpo2vec_random_walk` |
| `aggregation` | `ic_weighted_average` |
| `uses_ic_values` | `true` |
| `uses_dense_embeddings` | `true` |
| `uses_direct_hpo_overlap_for_score` | `false` |
| `score_note` | Explains cosine over IC-weighted averaged Word2Vec embeddings |
| `interpretation_note` | Clarifies matched terms are descriptive, not the score driver |
| `n_patient_terms_in_vocab` | How many patient terms had a Word2Vec embedding (diagnostics) |

---

## Run Statistics

The HPO2Vec+ pipeline records:

```text
number of raw patient terms
number of propagated patient terms used
number of diseases scored
number of diseases skipped (no embedding could be built)
runtime
```
