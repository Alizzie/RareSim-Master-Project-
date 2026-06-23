"""
Semantic similarity pipeline.

Implements IC-based pairwise HPO similarity using Best Match Average (BMA):
- Resnik BMA
- Lin BMA
- Jiang-Conrath BMA
"""

from raresim.core.context import AppContext
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.semantic.config import (
    ALL_METHODS,
    BMA_METHODS,
    PIPELINE_NAME,
    SEMANTIC_DIR,
)
from raresim.similarity_methods.semantic.explanation import build_explanation
from raresim.similarity_methods.semantic.methods import best_match_scores
from raresim.types.result import MethodResults, RunStats, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.hpo_utils import (
    filter_terms_by_ic,
    preprocess_ancestor_sets,
)
from raresim.utils.timer import Timer


def _run_bma_method(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    method_name: str,
    similarity_fn,
    patient_terms: set[str],
    all_patient_terms_before_filter: set[str],
    patient_raw_terms: set[str],
    config: PipelineConfig,
    ctx: AppContext,
    ancestor_sets: dict,
) -> tuple[list[SimilarityResult], RunStats]:
    """
    Run one BMA method over all disease profiles.

    For each disease:
    1. Filter disease terms by IC threshold.
    2. Compute bidirectional best match scores.
    3. Average patient-to-disease and disease-to-patient scores.
    4. Build an IC-based explanation.
    5. Attach category/profile metadata for display.
    """
    results = []
    skipped = 0
    method_timer = Timer(method_name).start()

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = filter_terms_by_ic(
            set(profile.get(config.terms_key, [])),
            ctx.ic_values,
            config.ic_threshold,
        )

        if not disease_terms:
            skipped += 1
            continue

        p2d_avg, p2d_matches = best_match_scores(
            patient_terms,
            disease_terms,
            ancestor_sets,
            ctx.ic_values,
            similarity_fn,
        )
        d2p_avg, d2p_matches = best_match_scores(
            disease_terms,
            patient_terms,
            ancestor_sets,
            ctx.ic_values,
            similarity_fn,
        )
        score = 0.5 * (p2d_avg + d2p_avg)

        explanation = build_explanation(
            method_name=method_name,
            patient_terms=patient_terms,
            disease_terms=disease_terms,
            score=score,
            p2d_avg=p2d_avg,
            d2p_avg=d2p_avg,
            p2d_matches=p2d_matches,
            d2p_matches=d2p_matches,
            all_patient_terms_before_filter=all_patient_terms_before_filter,
            hpo_labels=ctx.hpo_labels,
            ic_values=ctx.ic_values,
            ic_threshold=config.ic_threshold,
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
        n_patient_terms_propagated=len(all_patient_terms_before_filter),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=skipped,
        computation_time=method_timer.stop(),
    )

    return results, stats


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """
    Run selected BMA semantic similarity methods for the given patient.

    Returns:
        Dictionary mapping method name to MethodResults.
    """
    patient_raw_terms = set(patient.hpo_terms)
    patient_terms_before_filter = set(patient.get_terms(config.use_propagated_terms))
    patient_terms = filter_terms_by_ic(
        patient_terms_before_filter,
        ctx.ic_values,
        config.ic_threshold,
    )

    if not patient_terms:
        print("[semantic] Warning: no patient terms remain after IC filtering.")
        return {}

    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    all_results: dict[str, MethodResults] = {}

    for method_name, similarity_fn in BMA_METHODS.items():
        if method_name not in selected:
            continue

        results, stats = _run_bma_method(
            method_name=method_name,
            similarity_fn=similarity_fn,
            patient_terms=patient_terms,
            all_patient_terms_before_filter=patient_terms_before_filter,
            patient_raw_terms=patient_raw_terms,
            config=config,
            ctx=ctx,
            ancestor_sets=ancestor_sets,
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
    """Main entry point for running the semantic similarity pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=ALL_METHODS,
        run_fn=run,
        output_dir=SEMANTIC_DIR,
    )


if __name__ == "__main__":
    main()
