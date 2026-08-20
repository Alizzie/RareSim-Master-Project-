# Output

Everything RareSim writes to disk lives under `OUTPUTS_DIR` (`RARESIM_ROOT/outputs/` — see [Configuration](configuration.md)). This page is a map of what gets written where and by which layer, so you know which directory to check for a given question.

The real top-level list, confirmed against an actual `outputs/` tree, is longer than the layers documented in depth below:

```text
outputs/
├── artifacts/                      built ontology/profile artifacts (offline build phase) — documented below
├── similarity_methods/
│   └── cache/                      multi-method run caches (core/cache.py) — documented below
├── evaluation/                     per-dataset batch-runner cache + evaluator reports — documented below
├── evaluation_visual_questions/     benchmark visualization figures, CSVs, HTML report — documented below
├── validation_tools/                external validation-tool result files, by tool — documented below
└── webapp/                           output from the frontend/GUI
```


## `outputs/artifacts/` — the offline build output

Written once by `build/build_shared_artifacts.py` (see [Overview](overview.md#end-to-end-data-flow) and [Installation](../getting-started/installation.md)), and re-read by every runtime process via `AppContext.load()`. Nothing at runtime writes here — treat it as read-only build output that gets regenerated when ontology sources change.

```text
canonical_disease_profiles.json     main ORPHA-preferred profile set used for disease retrieval
disease_profiles.json               expanded profile set with aliases (OMIM/MONDO/etc.) as additional keys
hpo_labels.json                     HPO ID -> label map (display, term validation, prompts, text conversion)
hpo_parents.json                    direct HPO parent graph
hpo_ancestors.json                  full HPO ancestor closure (propagation, semantic similarity)
disease_parents.json                direct ORDO disease/category parent graph
disease_ancestors.json              ordered ORDO category chains (used for category paths in results)
disease_metadata_index.json         disease/category display metadata for ORDO IDs
term_frequencies.json               how many disease profiles contain each HPO term
information_content.json            IC values (semantic methods, IC-based filtering)
example_patient.json                sample PatientProfile artifact for demos/testing
orpha_mapping_index.json            alias-to-ORPHA mapping built from metadata xrefs/exact matches
alias_to_canonical.json             maps source-specific disease IDs to canonical IDs (used by the evaluator's
                                     alias matching — see evaluator-and-metrics.md)
canonical_filter_stats.json / expanded_filter_stats.json
                                     build diagnostics: how many profiles were filtered out and why
annotation_source_counts.json       build diagnostics: record counts per ontology/annotation source
term_provenance.json                per disease-term pair: which source/frequency was selected, and all candidates
negative_terms_by_disease.json      disease-HPO terms explicitly recorded as negative/excluded
```

`canonical_disease_profiles.json` vs. `disease_profiles.json` matters beyond the build phase: `PipelineConfig.use_canonical_profiles` (see [Configuration](configuration.md#per-run-configuration--pipelineconfig)) decides which one `AppContext` loads for a given run, which in turn affects whether OMIM/MONDO-style ground-truth IDs match during evaluation.


## `outputs/similarity_methods/cache/` — multi-method run caches

Written by `core/cache.py` (`save_run_cache()`), read back by `load_run_cache()` / `list_cached_runs()` / `print_cached_runs()`. This is separate from the evaluation cache below — it's for caching a full multi-method run for one patient (e.g. for the API/frontend to show a comparison view without recomputing every method), not for batch benchmark evaluation.

```json
{
  "schema_version": "1.2",
  "patient_id": "...",
  "run_timestamp": "...",
  "config": { "...": "PipelineConfig.to_dict()" },
  "app_metadata": { "...": "AppMetadata.to_dict()" },
  "methods_run": ["set_cosine", "semantic_resnik_bma", "..."],
  "results": { "...": "one MethodResults per method" }
}
```

Cache files are named from `patient_id` plus a timestamp, so re-running the same patient doesn't overwrite earlier runs.


## `outputs/evaluation/` — batch evaluation cache and reports

This is the batch-runner and evaluator output covered in full in the Evaluation section of this wiki — see [cache-format.md](../evaluation/cache-format.md) for the per-case cache schema and [evaluator-and-metrics.md](../evaluation/evaluator-and-metrics.md) for what the evaluator's four output files (`_evaluation.json`, `_evaluation_summary.txt`, `_stats.txt`, `_summary.tsv`) contain.

## `outputs/evaluation_visual_questions/` — benchmark visualization output

The figures, CSV tables, and self-contained HTML report produced by the benchmark visualization toolkit, built from everything under `outputs/evaluation/` (and optionally `outputs/validation_tools/`). Covered in full in [visualizing-results.md](../evaluation/visualizing-results.md#outputs).

## `outputs/validation_tools/` — external tool comparison results

Result files from running external validation tools (LIRICAL, Phenomizer, PhenoBrain, Dx29) as comparison baselines, produced by the runners under `scripts/validation_tools/`. The path convention (`<tool>_benchmarks/<dataset>_summary.tsv`) and how these feed into the visualization report are documented in [visualizing-results.md](../evaluation/visualizing-results.md#validation-tool-path-convention-drives-q5).


## The standard method output object — `MethodResults`

Regardless of which of the three output locations above a result ends up in, every similarity method produces the same standard shape (`types/result.py`), so evaluation, the API, and the frontend can all consume any method's output identically:

```text
SimilarityResult   one ranked disease: disease_id, label, score, method_name, rank,
                    aliases, category metadata, optional explanation
MethodResults       the full output for one method run: method name, pipeline name,
                    PipelineConfig used, RunStats observed during the run, the ranked
                    SimilarityResult list, and SCHEMA_VERSION
RunStats             what actually happened during the run: raw/propagated patient term
                    counts, terms actually used, diseases scored/skipped, elapsed time
                    (as opposed to PipelineConfig, which is what was requested)
```

`io.save_results(results, path)` writes a dict of `MethodResults` to one JSON file, keyed by a filesystem-safe method name. `io.save_individual_results(results, output_dir)` instead writes one file per method, named `<method>_top<k>.json` — useful when you want to inspect or diff one method's output without loading the others.
