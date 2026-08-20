# Denoising Autoencoder Similarity Method

## Overview

The denoising autoencoder (DAE) learns a compressed latent representation of HPO term sets by training to reconstruct clean binary vectors from deliberately corrupted inputs. At inference time, patient and disease profiles are encoded into their latent vectors and compared using Euclidean distance converted to a similarity score.

Unlike semantic or embedding methods that operate on individual term pairs or label text, the autoencoder works on the entire HPO profile at once as a fixed-length binary vector. The latent space captures co-occurrence structure across the whole vocabulary.

The method is implemented across four files:

```text
config.py        Architecture dimensions, training hyperparameters, cache paths
methods.py       Vocabulary, binary vectors, corruption, network definition, euclidean similarity
pipeline.py      Pipeline entry point, model training/loading, encoding loop
explanation.py   Method-specific explanation block and base explainer delegation
```


## High-Level Method Logic

The autoencoder pipeline follows this sequence:

```text
Disease profiles
        ↓
Build vocabulary (sorted union of all propagated HPO terms)
        ↓
Convert each disease profile to a binary vector of length vocab_size
        ↓
Train denoising autoencoder on those binary vectors
        ↓
For each patient:
    Convert patient HPO terms to binary vector
        ↓
    Encode patient vector → latent representation
        ↓
    For each disease:
        Encode disease vector → latent representation
            ↓
        Euclidean similarity between patient and disease latent vectors
            ↓
    Rank diseases by score
```


## Step 1: Build the Vocabulary

A sorted list of all unique HPO terms across all disease profiles is collected from propagated terms:

```python
vocab = sorted(union of all propagated_hpo_terms across diseases)
```

This is the fixed-dimension binary vector space. Every term is assigned an index. The vocabulary size determines the input and output dimensions of the autoencoder.


## Step 2: Binary Vectors

Each profile is converted to a binary vector of length `vocab_size`:

```
vec[i] = 1.0  if vocab[i] in profile_terms
vec[i] = 0.0  otherwise
```

No IC weighting at this stage. The autoencoder learns to weight terms implicitly through training by observing which terms co-occur across disease profiles.


## Step 3: Train the Autoencoder

### Architecture

A 4-layer fully connected network:

```
Input (vocab_size)
    → H1 (hidden_dim=512, ReLU)
    → Latent (latent_dim=128, ReLU)
    → H3 (hidden_dim=512, Sigmoid)
    → Output (vocab_size, Sigmoid)
```

The encoder uses **ReLU** activations with **He initialization** (`√(2 / fan_in)`). The decoder uses **Sigmoid** activations with **Xavier initialization** (`√(2 / (fan_in + fan_out))`). This combination avoids vanishing gradients in the bottleneck while keeping decoder outputs in [0, 1] for BCE loss.

### Corruption

During training each clean vector is corrupted before being passed to the encoder:

**Masking noise** (`NOISE_RATE = 0.2`): 20% of present terms are randomly dropped to zero.

**False positive injection** (`FALSE_POSITIVE_RATE = 0.05`): 5% of absent terms are randomly set to one.

The model must reconstruct the clean vector from the corrupted input. Masking alone could be solved by the trivial solution of outputting zeros everywhere; false positive injection forces the model to learn the true co-occurrence structure and distinguish genuine from injected term presence.

### Loss and Optimizer

- **Loss**: Binary Cross-Entropy between the clean input vector and the reconstruction
- **Optimizer**: SGD with momentum (`lr = 0.01`, `momentum = 0.9`)

### Training Parameters

| Parameter | Value |
|-----------|-------|
| `HIDDEN_DIM` | 512 |
| `LATENT_DIM` | 128 |
| `LEARNING_RATE` | 0.01 |
| `MOMENTUM` | 0.9 |
| `NOISE_RATE` | 0.2 |
| `FALSE_POSITIVE_RATE` | 0.05 |
| `EPOCHS` | 50 |
| `BATCH_SIZE` | 64 |


## Step 4: Encode Profiles

At inference time, corruption is **not applied**. Clean binary vectors are passed through the encoder only:

```
h1     = ReLU(x @ W1 + b1)
latent = ReLU(h1 @ W2 + b2)
```

This produces a 128-dimensional latent vector for each patient and disease. The decoder is not used at inference.


## Step 5: Euclidean Similarity

L2 distance between latent vectors is converted to a bounded similarity score:

```
distance   = ||patient_latent − disease_latent||₂
similarity = 1 / (1 + distance)
```

- Score = 1.0 → identical latent vectors
- Score → 0 → very distant vectors
- Score is always in **(0, 1]**

Euclidean distance is used instead of cosine similarity because the latent space uses ReLU activations, which produce non-negative sparse vectors where cosine similarity is less reliable — two vectors can appear similar in direction while being very different in magnitude.


## Model Caching

Trained models are saved to `model_cache/` as `.npz` files containing all weight matrices and biases. The model is loaded on subsequent runs without retraining. Delete the cache to force retraining.


## Important: Score vs HPO Overlap

The score is driven entirely by **latent space geometry**, not by direct HPO term overlap. Matched and unmatched term fields in the explanation are computed from raw HPO set intersection for readability only and do not affect the score. The explanation explicitly notes this.


## Explanation Fields

| Field | Description |
|-------|-------------|
| `summary` | Score, direct overlap count, and clarification that score is latent-based |
| `coverage` | Patient/disease % matched, term counts (descriptive only) |
| `matched_terms` | HPO terms shared between patient and disease (descriptive only) |
| `unmatched_patient_terms` | Patient terms not in the disease (descriptive only) |
| `embedding_method` | `denoising_autoencoder_latent` |
| `aggregation` | `binary_vocab_vector` |
| `uses_ic_values` | `false` |
| `uses_dense_embeddings` | `true` |
| `uses_direct_hpo_overlap_for_score` | `false` |
| `score_note` | Explains Euclidean similarity over encoded latent vectors |
| `interpretation_note` | Clarifies matched terms are descriptive, not the score driver |


## Run Statistics

The autoencoder pipeline records:

```text
number of raw patient terms
number of propagated patient terms used
number of diseases scored
number of diseases skipped (no terms in vocabulary)
runtime
```


## Note on Sparse Patient Profiles

Performance degrades with very sparse patient profiles (fewer than ~10 HPO terms) because the binary input vector becomes too sparse for the latent space to capture meaningful structure. The method works best when the patient has a reasonably complete phenotype description. This is shown as a warning note in the UI method selector.
