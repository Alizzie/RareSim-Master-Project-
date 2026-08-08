# Implementation and Libraries

## Purpose

A quick reference for what RareSim is actually built with — languages, key libraries, and the tools/frameworks tying the pieces together.

## Programming Language

Python, across all four packages except one:

```text
raresim-core        Python
raresim-backend      Python
raresim-cli           Python
scripts (evaluation)  Python
raresim-frontend      JavaScript/Vue
```

## Libraries

**Transformer and LLM methods:** Hugging Face `transformers` and `sentence-transformers`, backed by PyTorch — used to load and run the biomedical encoders (SapBERT, BioBERT, ClinicalBERT, PubMedBERT, MiniLM) and the LLM (Mistral-7B-Instruct-v0.2). See [Deployment and External Dependencies](/system/deployment-and-external-dependencies#hugging-face-model-downloads) for the loading mechanics and model identifiers.

**GPT-based HPO extraction:** the `openai` client library, against GPT-4o-mini.

## Tools and Frameworks

```text
Backend               FastAPI service
Frontend               Vue single-page application, served by Vite in dev
Persistent storage      JSON (or JSON Lines) files on local disk,
                       throughout the whole system — no database
LIRICAL / Phenomizer   Invoked as local Java processes via subprocess,
                       not through any Python library
PhenoBrain / Dx29        Called as HTTP APIs
Compute-heavy methods    LLM and transformer methods run on GPU
(LLM, transformer)      infrastructure: anna.ifi.uzh.ch, george.ifi.uzh.ch
```

```text
Backend       FastAPI, Pydantic, Uvicorn
Frontend       Vue 3, Vite, vue-router
Testing        pytest
```
