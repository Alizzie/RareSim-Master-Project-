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

from raresim.core.context import AppContext
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.set_based.config import (
    METHOD_MAP,
    PIPELINE_NAME,
    SETBASED_DIR,
)
from raresim.similarity_methods.set_based.explanation import build_explanation
from raresim.types.result import MethodResults, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.timer import Timer


def run(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the set-based similarity pipeline for the given patient."""
    patient_raw_terms = set(patient.hpo_terms)
    patient_terms = set(patient.get_terms(config.use_propagated_terms))

    all_results: dict[str, MethodResults] = {}

    for method_name, similarity_fn in METHOD_MAP.items():
        if method_name not in selected:
            continue

        results = []
        n_skipped = 0
        method_timer = Timer(method_name).start()

        for disease_id, profile in ctx.disease_profiles.items():
            disease_terms = set(profile.get(config.terms_key, []))

            if not disease_terms:
                n_skipped += 1
                continue

            score = similarity_fn(patient_terms, disease_terms)

            explanation = build_explanation(
                method_name=method_name,
                patient_terms=patient_terms,
                disease_terms=disease_terms,
                score=score,
                hpo_labels=ctx.hpo_labels,
                ic_values=ctx.ic_values,
                patient_raw_terms=patient_raw_terms,
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
            n_patient_terms_raw=len(patient_raw_terms),
            n_patient_terms_propagated=len(
                patient.get_terms(use_propagated=True)
            ),
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
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=list(METHOD_MAP.keys()),
        run_fn=run,
        output_dir=SETBASED_DIR,
    )


if __name__ == "__main__":
    main()
