"""
Set-based similarity pipeline.

Uses:
- Jaccard
- Dice
- Overlap Coefficient
- Cosine similarity over binary HPO term sets

Explanation:
- delegated to similarity_methods/set_based/explanation.py
"""

from raresim.core import (
    AppContext,
    run_similarity_method,
    build_run_stats,
    sort_and_rank,
)
from raresim.ontology import build_category_metadata
from raresim.similarity_methods.set_based.config import (
    METHOD_MAP,
    PIPELINE_NAME,
    SETBASED_DIR,
    METHODS_REQUIRING_EXCLUSIONS,
    NEGATIVE_PENALTY_WEIGHT,
)
from raresim.similarity_methods.set_based.explanation import build_explanation
from raresim.types import (
    MethodResults,
    SimilarityResult,
    PipelineConfig,
    PatientProfile,
)
from raresim.utils.timer import Timer
from raresim.utils.disease_profile_utils import disease_exclusion_inputs


def run(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the set-based similarity pipeline for the given patient."""
    patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_excluded_terms = patient.get_excluded_terms()

    all_results: dict[str, MethodResults] = {}

    for method_name, similarity_fn in METHOD_MAP.items():
        if method_name not in selected:
            continue

        results = []
        n_skipped = 0
        method_timer = Timer(method_name).start()
        needs_exclusions = method_name in METHODS_REQUIRING_EXCLUSIONS

        for disease_id, profile in ctx.disease_profiles.items():
            disease_terms = set(profile.get(config.terms_key, []))
            disease_raw_terms, disease_excluded_terms = disease_exclusion_inputs(
                profile
            )

            if not disease_terms:
                n_skipped += 1
                continue

            if needs_exclusions:
                score = similarity_fn(
                    patient_terms,
                    disease_terms,
                    patient_excluded_terms,
                    disease_excluded_terms,
                    penalty_weight=NEGATIVE_PENALTY_WEIGHT,
                )
            else:
                score = similarity_fn(patient_terms, disease_terms)

            explanation = build_explanation(
                method_name=method_name,
                patient_terms=patient_terms,
                disease_terms=disease_terms,
                disease_excluded_terms=disease_excluded_terms,
                disease_raw_terms=disease_raw_terms,
                score=score,
                hpo_labels=ctx.hpo_labels,
                ic_values=ctx.ic_values,
                patient=patient,
            )

            category_metadata = build_category_metadata(
                disease_id=disease_id,
                profile=profile,
                disease_ancestors=ctx.disease_ancestors,
                disease_metadata_index=ctx.disease_metadata_index,
            )

            results.append(
                SimilarityResult(
                    disease_id=disease_id,
                    label=profile.get("label", ""),
                    profile_type=category_metadata["profile_type"],
                    category_source_id=category_metadata["category_source_id"],
                    category_path=category_metadata["category_path"],
                    matched_aliases=category_metadata["matched_aliases"],
                    score=score,
                    method_name=method_name,
                    explanation=explanation.to_dict(),
                )
            )

        stats = build_run_stats(
            n_patient_terms_raw=len(patient.hpo_terms),
            n_patient_terms_propagated=len(patient.get_terms(use_propagated=True)),
            n_patient_terms_used=len(patient_terms),
            n_diseases_scored=len(results),
            n_diseases_skipped=n_skipped,
            computation_time=method_timer.stop(),
        )

        all_results[method_name] = sort_and_rank(
            results,
            config,
            stats,
            method_name,
            PIPELINE_NAME,
        )

    return all_results


def main() -> None:
    """Run the set-based similarity pipeline."""
    run_similarity_method(
        pipeline_name=PIPELINE_NAME,
        method_names=list(METHOD_MAP.keys()),
        run_fn=run,
        output_dir=SETBASED_DIR,
    )


if __name__ == "__main__":
    main()
