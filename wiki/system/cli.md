# CLI

## Purpose

`raresim-cli` is the third way to run RareSim's retrieval pipeline, alongside the [backend API](/system/web-interface), the [frontend](/system/web-interface), and the offline [batch runners](/evaluation/workflow-overview). It's a terminal application (`app.py`) for direct, scripted, or interactive use without a running API server.

## Patient Input Modes

Five ways to supply a patient case:

```text
1. Raw clinical text     Extracted via the same build_patient_profile()
                          used elsewhere — see hpo_extraction.ensemble's
                          builder on Patient Profile Construction.
2. HPO term list         Comma-separated HPO term IDs, passed directly.
3. Patient JSON file      A pre-built patient JSON file.
4. Example patient        Built-in fixture (the same EXAMPLE_PATIENT
                          used to generate example_patient.json — see
                          File Reference and Runtime Loading).
5. Interactive prompt      Terminal prompt, used when no input flag is
                          given.
```

Method selection follows the same pattern: `--methods`, defaulted to all methods, or chosen interactively.

## Dispatch

Dispatch to the core pipeline is structurally identical to the backend's — the CLI loops over the same `METHOD_MODULES` registry and calls:

```python
module.run(patient, selected, config, ctx)
```

for each selected pipeline. This is the same registry and the same `run(patient, methods, config, ctx)` interface documented in [Web Interface](/system/web-interface#method-dispatch) and [Architecture Design](/system/architecture-design#similarity-method-architecture) — the CLI is simply another caller of the identical dispatch pattern, not a separate implementation of it.

## Output

Results are persisted **twice**:

1. Via `save_run_cache()` — the same runtime cache used by interactive API and frontend runs.
2. Via CLI-specific convenience outputs written to `outputs/raresim_cli/`: combined and per-method result files, plus a disease co-occurrence summary and timing summary, built by `_summary.py`.
