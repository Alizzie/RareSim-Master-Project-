# Welcome to the RareSim wiki!

RareSim retrieves and ranks candidate rare diseases for a patient, given either structured HPO phenotype terms or raw clinical text. It implements several independent similarity methods against a shared corpus of disease profiles, and provides shared infrastructure.

New here? Start with [Installation](getting-started/installation.md) and [Quick Start](getting-started/quick-start.md). If you want to read more about how the project is put together, head to [Project Overview](project-overview/overview.md).

## Project Overview

- [Overview](project-overview/overview.md)
- [Configuration](project-overview/configuration.md)
- [Output](project-overview/output.md)

## Getting Started

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)

## System

- [Architecture & Design](system/architecture-design.md)
- [Web Interface](system/web-interface.md)
- [Web Interface Demo](system/web-interface-demo.md)
- [CLI](system/cli.md)
- [HPO Extraction Pipeline](system/hpo-extraction-pipeline.md)
- [Data & Storage Lifecycle](system/data-and-storage-lifecycle.md)
- [Deployment & External Dependencies](system/deployment-and-external-dependencies.md)
- [Implementation & Libraries](system/implementation-and-libraries.md)

## Artifacts

How the offline build phase turns raw ontology sources into the shared profile/artifact files every similarity method reads.

- [Overview](artifacts/shared-overview.md)
- [Full Workflow](artifacts/full-workflow.md)
- [Raw Sources & Ontology Loading](artifacts/raw-sources-and-ontology-loading.md)
- [Disease ID Normalization & Mapping](artifacts/disease-id-normalization-and-mapping.md)
- [Disease Profile Construction](artifacts/disease-profile-construction.md)
- [Patient Profile Construction](artifacts/patient-profile-construction.md)
- [File Reference & Runtime Loading](artifacts/file-reference-and-runtime-loading.md)

## Similarity Methods

- [Overview](similarity-methods/overview.md)
- [TF-IDF](similarity-methods/tfidf-methods.md)
- [Semantic](similarity-methods/semantic-methods.md)
- [Set-Based](similarity-methods/set-based-methods.md)
- [HPO2Vec+](similarity-methods/hpo2vec.md)
- [Denoising Autoencoder](similarity-methods/denoising-autoencoder.md)
- [Embedding methods](similarity-methods/embedding.md)
- [LLM](similarity-methods/llm.md)
- [Adding a New Method](similarity-methods/adding-new-method.md)

## Evaluation

- [Workflow Overview](evaluation/workflow-overview.md)
- [Dataset Format](evaluation/dataset-format.md)
- [Available Datasets](evaluation/dataset-available.md)
- [Adding a Dataset](evaluation/dataset-adding.md)
- [Benchmark Dataset Standardization](evaluation/benchmark-dataset-standardization.md)
- [Batch Runners & Shared Utilities](evaluation/batch-runners-and-shared-utilities.md)
- [Cache Format](evaluation/cache-format.md)
- [Evaluator & Metrics](evaluation/evaluator-and-metrics.md)
- [Visualizing Results](evaluation/visualizing-results.md)
- [Adding a Method (evaluation runner)](evaluation/adding-method.md)

## Validation Tools

External diagnostic tools run as comparison baselines against the same benchmark datasets.

- [Tools Overview](validation/tools-overview.md)
- [LIRICAL](validation/lirical.md)
- [Phenomiser](validation/phenomiser.md)
- [PhenoBrain](validation/phenobrain.md)
- [PhenoBrain (Local)](validation/phenobrain-local.md)
- [DX29 Search](validation/dx29-search.md)
- [DX29 Phrank](validation/dx29-phrank.md)
- [Output](validation/output.md)
