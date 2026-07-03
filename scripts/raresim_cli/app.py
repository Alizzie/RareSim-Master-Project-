"""
Terminal interface for the rare disease similarity pipeline.

Two input modes:
    --text   Raw clinical text → extract HPO terms → run similarity
    --hpo    Pre-extracted HPO terms (comma-separated or JSON) → run similarity

All output is written to outputs/raresim_cli/.

Usage:
    python -m raresim_cli.app --text "Patient with cerebellar ataxia."
    python -m raresim_cli.app --hpo HP:0001251,HP:0000256
    python -m raresim_cli.app --patient path/to/patient.json
    python -m raresim_cli.app --defaults
    python -m raresim_cli.app --text "..." --methods semantic_resnik_bma tfidf_hpo
    python -m raresim_cli.app --defaults --top-k 5
"""

from raresim.core.cache import save_run_cache
from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.similarity_methods.registry import (
    ALL_METHODS,
    METHOD_MODULES,
    DEFAULTS,
)
from raresim.types.result import MethodResults
from raresim.utils.io import load_json, save_json
from raresim.utils.paths import (
    HPO_LABELS_PATH,
)

from _cli_parser import parse_args
import _utils as gu
import _summary as gsum
from _patient_input import resolve_patient
from _utils import RARESIM_CLI_DIR


def _resolve_selected_methods(args, prompt_fn) -> list[str]:
    """Determine which methods to run from args or interactive prompt."""
    if args.methods:
        return args.methods
    if args.defaults:
        return ALL_METHODS
    return prompt_fn(ALL_METHODS)


def _run_selected_pipelines(
    selected: list[str],
    patient,
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """
    Run every pipeline that has at least one selected method.

    All pipelines now share the run(patient, selected, config, ctx) signature
    and return dict[str, MethodResults], so dispatch is a uniform loop over
    the METHOD_MODULES registry — no per-pipeline special-casing.
    """
    all_results: dict[str, MethodResults] = {}

    for pipeline_name, module in METHOD_MODULES.items():
        # Skip pipelines none of whose methods were selected
        if not any(m in selected for m in module.METHOD_NAMES):
            continue
        print(f"\n  Running {pipeline_name} methods...")
        all_results.update(module.run(patient, selected, config, ctx))

    return all_results


def main() -> None:
    """Terminal interface entry point."""
    gu.check_artifacts_exist()
    args = parse_args()

    RARESIM_CLI_DIR.mkdir(parents=True, exist_ok=True)
    hpo_labels = load_json(HPO_LABELS_PATH)

    print("=" * 64)
    print("  Rare Disease Similarity Pipeline")
    print("=" * 64)

    # ── Resolve patient (all input-mode branching lives in _patient_input) ────
    patient = resolve_patient(args, hpo_labels, prompt_fn=gu.prompt_patient)
    print(f"  HPO terms       : {len(patient.hpo_terms)}")
    print(f"  Propagated terms: {len(patient.propagated_hpo_terms)}")

    # ── Methods ───────────────────────────────────────────────────────────────
    selected = _resolve_selected_methods(args, gu.prompt_methods)
    print(f"\nSelected methods ({len(selected)}): {', '.join(selected)}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = PipelineConfig(
        top_k=args.top_k,
        use_propagated_terms=not args.no_propagation,
        ic_threshold=args.ic_threshold,
        use_canonical_profiles=DEFAULTS["use_canonical_profiles"],
    )
    print(
        f"Config: top_k={config.top_k}, "
        f"propagated={config.use_propagated_terms}, "
        f"ic_threshold={config.ic_threshold}"
    )

    # ── Shared context ────────────────────────────────────────────────────────
    print("\nLoading shared context...")
    ctx = AppContext.load(patient, config.use_canonical_profiles)
    gu.print_app_metadata(ctx.app_metadata)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\nRunning pipeline...")
    all_results = _run_selected_pipelines(
        selected=selected,
        patient=patient,
        config=config,
        ctx=ctx,
    )

    # ── Display ───────────────────────────────────────────────────────────────
    for method_results in all_results.values():
        gu.print_results_table(method_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    gu.save_results(all_results, ctx.app_metadata)
    save_run_cache(
        patient_id=patient.patient_id,
        config=config,
        similarity_results=all_results,
        app_metadata=ctx.app_metadata,
    )

    # ── Summaries ─────────────────────────────────────────────────────────────
    disease_summary = gsum.build_disease_summary(all_results)
    timing_summary = gsum.build_timing_summary(all_results)
    gsum.print_disease_summary(disease_summary)
    gsum.print_timing_summary(timing_summary)
    save_json(disease_summary, RARESIM_CLI_DIR / "disease_summary.json")
    save_json(timing_summary, RARESIM_CLI_DIR / "timing_summary.json")


if __name__ == "__main__":
    main()
