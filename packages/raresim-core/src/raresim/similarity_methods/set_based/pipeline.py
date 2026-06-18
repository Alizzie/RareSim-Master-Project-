"""
Set-based similarity pipeline.

Uses: Jaccard, Dice, Overlap Coefficient, Cosine (binary).
Explanation: delegated to similarity_methods/set_based/explanation.py.
"""

from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.timer import Timer
from raresim.types.result import SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.core.context import AppContext
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.similarity_methods.set_based.config import (
    METHOD_MAP,
    PIPELINE_NAME,
    SETBASED_DIR,
)
from raresim.similarity_methods.set_based.explanation import build_explanation


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, list[SimilarityResult]]:
    """Run the set-based similarity pipeline for the given patient and selected methods."""

    patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_raw_terms = set(patient.hpo_terms)

    all_results = {}

    for method_name, fn in METHOD_MAP.items():
        if method_name not in selected:
            continue

        results = []
        n_skipped = 0
        timer = Timer(method_name).start()

        for disease_id, profile in ctx.disease_profiles.items():
            disease_terms = set(profile.get(config.terms_key, []))

            if not disease_terms:
                n_skipped += 1
                continue

            score = fn(patient_terms, disease_terms)

            explaination = build_explanation(
                method_name=method_name,
                patient_terms=patient_terms,
                disease_terms=disease_terms,
                score=score,
                hpo_labels=ctx.hpo_labels,
                ic_values=ctx.ic_values,
                patient_raw_terms=patient_raw_terms,
            )

            results.append(
                SimilarityResult(
                    disease_id=disease_id,
                    label=profile.get("label", ""),
                    score=score,
                    method_name=method_name,
                    explanation=explaination.to_dict(),
                )
            )

        stats = build_run_stats(
            n_patient_terms_raw=len(patient_raw_terms),
            n_patient_terms_propagated=len(patient.propagated_hpo_terms),
            n_patient_terms_used=len(patient_terms),
            n_diseases_scored=len(results),
            n_diseases_skipped=n_skipped,
            computation_time=timer.stop(),
        )

        all_results[method_name] = sort_and_rank(
            results, config, stats, method_name, PIPELINE_NAME
        )

    return all_results


def main() -> None:
    """Example main function to run the set-based similarity pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=list(METHOD_MAP.keys()),
        run_fn=run,
        output_dir=SETBASED_DIR,
    )


if __name__ == "__main__":
    main()
