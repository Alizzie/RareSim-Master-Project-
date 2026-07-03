"""
Starting point for running a similarity method pipeline.
"""

from pathlib import Path

from raresim.core.context import AppContext
from raresim.types import PatientProfile, PipelineConfig
from raresim.utils.io import save_results, save_individual_results
from raresim.utils.patient_loader import load_patient
from raresim.utils.paths import PATIENT_DIR
from raresim.utils.timer import timer


def run_similarity_method(
    pipeline_name: str,
    method_names: list[str],
    run_fn,
    output_dir: Path,
    config: PipelineConfig | None = None,
    patient: PatientProfile | None = None,
) -> None:
    """
    Run a single similarity method within a pipeline.
    Handles loading, running, saving, and printing results.

    Usage to call similarity method in each method separately:
        def main() -> None:
            run_similarity_method(
                pipeline_name="tfidf",
                method_names=[METHOD_NAME],
                run_fn=run,
                output_dir=OUTPUT_DIR,
            )
    """
    if config is None:
        config = PipelineConfig()

    if patient is None:
        patient = load_patient(PATIENT_DIR / "example_patient.json")

    print(f"[{pipeline_name}] Loaded patient: {patient.patient_id}")
    print(f"[{pipeline_name}] HPO terms: {patient.hpo_terms}")
    print(f"[{pipeline_name}] Propagated HPO Terms: {patient.propagated_hpo_terms}")
    print(f"[{pipeline_name}] Method Names: {method_names}")

    ctx = AppContext.load(
        patient=patient, use_canonical_profiles=config.use_canonical_profiles
    )

    print(f"[{pipeline_name}] Running pipeline with methods: {method_names}")
    with timer(f"full {pipeline_name} pipeline"):
        results = run_fn(patient, method_names, config, ctx)

    print(f"[{pipeline_name}] Pipeline finished. Saving results to {output_dir}...")
    save_results(
        results, output_dir / f"list_all_{pipeline_name}_top{config.top_k}.json"
    )
    save_individual_results(results, output_dir)
