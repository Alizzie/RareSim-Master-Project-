"""
Main script to run the semantic similarity pipeline, integrating disease profiles,
patient data, and cosine similarity calculations to rank diseases based on HPO term overlap.
"""

from shared.io import save_individual_results, load_patient
from shared.paths import PROJECT_ROOT, SHARED_DIR
from shared.pipeline_config import PipelineConfig
from shared.context import AppContext
from shared.result import Metadata, SimilarityResult
from core.schemas import PatientProfile
from similarity_methods.methods import (
    cosine_similarity,
    jaccard_similarity,
    overlap_coefficient,
    dice_similarity,
)
from similarity_methods.set_based.utils import extend_explaination
from similarity_methods.utils import sort_and_rank

SETBASED_DIR = PROJECT_ROOT / "outputs" / "set_based"

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
) -> dict:
    """Run the set-based similarity pipeline for the given patient and selected methods."""

    terms_key = "propagated_hpo_terms" if config.use_propagated_terms else "hpo_terms"
    patient_terms = set(
        patient.propagated_hpo_terms
        if config.use_propagated_terms
        else patient.hpo_terms
    )

    all_results = {}

    for method_name, fn in METHOD_MAP.items():
        if method_name not in selected:
            continue

        results = []
        for disease_id, profile in ctx.disease_profiles.items():
            disease_terms = set(profile.get(terms_key, []))
            score, explaination = fn(patient_terms, disease_terms)
            explaination = extend_explaination(
                explaination, patient_terms, disease_terms
            )

            results.append(
                SimilarityResult(
                    disease_id=disease_id,
                    label=profile.get("label", ""),
                    score=score,
                    method_name=method_name,
                    explanation=explaination,
                    metadata=Metadata(
                        method_name=method_name,
                        pipeline_name="set_based",
                        use_propagated_terms=config.use_propagated_terms,
                        ic_threshold=config.ic_threshold,
                        top_k=config.top_k,
                        n_patient_terms=len(patient_terms),
                        n_disease_terms=len(disease_terms),
                    ),
                )
            )

        all_results[method_name] = sort_and_rank(results, config.top_k)

    return all_results


def main() -> None:
    """Example main function to run the set-based similarity pipeline."""
    patient = load_patient(SHARED_DIR / "example_patient.json")
    config = PipelineConfig()
    ctx = AppContext.load(patient, config.use_canonical_profiles)

    results = run(patient, list(METHOD_MAP.keys()), config, ctx)

    for method_name, rows in results.items():
        save_individual_results(
            rows, SETBASED_DIR / f"{method_name}_top{config.top_k}.json"
        )


if __name__ == "__main__":
    main()
