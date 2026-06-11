"""
Standardized main() runner for all similarity pipelines.
Used by individual pipeline scripts, not part of the installable package.
"""

from pathlib import Path

from raresim.core.pipeline import PipelineConfig
from raresim.core.context import AppContext
from raresim.utils.io import save_results, save_individual_results
from raresim.utils.patient_loader import load_patient
from raresim.utils.paths import PATIENT_DIR


def run_pipeline_main(
    pipeline_name: str,
    method_names: list[str],
    run_fn,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> None:
    """
    Standardized main() for all pipelines.
    Handles loading, running, saving, and printing results.

    Usage in each pipeline:
        def main() -> None:
            run_pipeline_main(
                pipeline_name="tfidf",
                method_names=[METHOD_NAME],
                run_fn=run,
                output_dir=OUTPUT_DIR,
            )
    """
    if config is None:
        config = PipelineConfig()

    patient = load_patient(PATIENT_DIR / "example_patient.json")
    ctx = AppContext.load(
        patient=patient, use_canonical_profiles=config.use_canonical_profiles
    )

    results = run_fn(patient, method_names, config, ctx)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir / f"{pipeline_name}_top{config.top_k}.json")

    for method_name, rows in results.items():
        save_individual_results(
            rows, output_dir / f"{method_name}_top{config.top_k}.json"
        )
        print(f"\nTop results for {method_name}:")
        for r in rows:
            print(
                f"  rank={r.rank:>2} | {r.disease_id:<15} | score={r.score:.4f} | {r.label}"
            )
