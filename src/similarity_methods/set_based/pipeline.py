"""
Main script to run the semantic similarity pipeline, integrating disease profiles,
patient data, and cosine similarity calculations to rank diseases based on HPO term overlap.
"""

from shared.paths import PROJECT_ROOT
from shared.context import AppContext
from shared.result import SimilarityResult
from shared.pipeline import (
    PipelineConfig,
    build_metadata,
    sort_and_rank,
    run_pipeline_main,
)
from core.schemas import PatientProfile
from shared.methods import (
    cosine_similarity,
    jaccard_similarity,
    overlap_coefficient,
    dice_similarity,
)
from shared.explaination import expand, SET_BASED_EXPLANATION

SETBASED_DIR = PROJECT_ROOT / "outputs" / "set_based"
PIPELINE_NAME = "set_based"

METHOD_MAP = {
    "set_cosine": cosine_similarity,
    "set_jaccard": jaccard_similarity,
    "set_overlap": overlap_coefficient,
    "set_dice": dice_similarity,
}


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, list[SimilarityResult]]:
    """Run the set-based similarity pipeline for the given patient and selected methods."""

    patient_terms = set(patient.get_terms(config.use_propagated_terms))

    all_results = {}

    for method_name, fn in METHOD_MAP.items():
        if method_name not in selected:
            continue

        results = []
        for disease_id, profile in ctx.disease_profiles.items():
            disease_terms = set(profile.get(config.terms_key, []))
            score, explaination = fn(patient_terms, disease_terms)
            explaination = expand(
                explaination, patient_terms, disease_terms, SET_BASED_EXPLANATION
            )

            results.append(
                SimilarityResult(
                    disease_id=disease_id,
                    label=profile.get("label", ""),
                    score=score,
                    method_name=method_name,
                    explanation=explaination,
                    metadata=build_metadata(
                        method_name=method_name,
                        pipeline_name=PIPELINE_NAME,
                        config=config,
                        n_patient_terms=len(patient_terms),
                        n_disease_terms=len(disease_terms),
                        ctx=ctx,
                    ),
                )
            )

        all_results[method_name] = sort_and_rank(results, config.top_k)

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
