"""
Semantic similarity pipeline
Implements IC-based pairwise HPO similarity using Best Match Average (BMA):
- Resnik BMA
- Lin BMA
- Jiang-Conrath BMA
"""

from core.schemas import PatientProfile
from shared.context import AppContext
from shared.paths import PROJECT_ROOT
from shared.result import SimilarityResult
from shared.pipeline import (
    PipelineConfig,
    build_metadata,
    sort_and_rank,
    run_pipeline_main,
)
from shared.explaination import expand, SEMANTIC_EXPLANATION
from shared.math import (
    filter_terms_by_ic,
    preprocess_ancestor_sets,
)
from shared.methods import best_match_scores

from similarity_methods.semantic.methods import (
    resnik_similarity,
    lin_similarity,
    jiang_conrath_similarity,
)

SEMANTIC_DIR = PROJECT_ROOT / "outputs" / "semantic"
PIPELINE_NAME = "semantic"

# BMA methods: pairwise term-to-term comparison averaged bidirectionally
BMA_METHODS = {
    "semantic_resnik_bma": resnik_similarity,
    "semantic_lin_bma": lin_similarity,
    "semantic_jiang_conrath_bma": jiang_conrath_similarity,
}

ALL_METHODS = list(BMA_METHODS)


def _run_bma_method(
    method_name: str,
    similarity_fn,
    patient_terms: set,
    config: PipelineConfig,
    ctx: AppContext,
    ancestor_sets: dict,
) -> list[SimilarityResult]:
    """
    Run one BMA method over all disease profiles.

    For each disease:
    1. Filter disease terms by IC threshold
    2. Compute bidirectional best match scores (patient→disease, disease→patient)
    3. Average the two directions → final BMA score
    4. Build explanation with BMA averages + shared coverage expanders
    """
    results = []
    skipped = 0

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
            patient_terms, disease_terms, ancestor_sets, ctx.ic_values, similarity_fn
        )
        d2p_avg, d2p_matches = best_match_scores(
            disease_terms, patient_terms, ancestor_sets, ctx.ic_values, similarity_fn
        )
        score = 0.5 * (p2d_avg + d2p_avg)

        explanation = {
            "method": method_name,
            "patient_to_disease_avg": p2d_avg,
            "disease_to_patient_avg": d2p_avg,
            "ic_threshold_used": config.ic_threshold,
            "n_patient_terms_after_ic_filter": len(patient_terms),
            "n_disease_terms_after_ic_filter": len(disease_terms),
            "top_patient_to_disease_matches": sorted(
                p2d_matches, key=lambda x: x["score"], reverse=True
            )[:5],
            "top_disease_to_patient_matches": sorted(
                d2p_matches, key=lambda x: x["score"], reverse=True
            )[:5],
        }

        # add shared coverage expanders (shared_terms, coverage, term_counts, unmatched)
        explanation = expand(explanation, patient_terms, disease_terms, SEMANTIC_EXPLANATION)

        results.append(
            SimilarityResult(
                disease_id=disease_id,
                label=profile.get("label", ""),
                score=score,
                method_name=method_name,
                explanation=explanation,
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

    return results


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, list[SimilarityResult]]:
    """
    Run selected BMA semantic similarity methods for the given patient.

    Args:
        patient:  Patient profile with HPO terms.
        selected: List of method names to run (subset of ALL_METHODS).
        config:   Pipeline configuration (top_k, ic_threshold, etc.).
        ctx:      AppContext with disease profiles, IC values, ancestors.

    Returns:
        Dict mapping method_name → ranked list of SimilarityResult.
    """
    patient_terms = filter_terms_by_ic(
        set(patient.get_terms(config.use_propagated_terms)),
        ctx.ic_values,
        config.ic_threshold,
    )

    if not patient_terms:
        print("[semantic] Warning: no patient terms remain after IC filtering.")
        return {}

    # preprocess ancestor sets once — reused across all BMA methods
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)

    all_results = {}

    for method_name, similarity_fn in BMA_METHODS.items():
        if method_name not in selected:
            continue
        results = _run_bma_method(
            method_name, similarity_fn, patient_terms, config, ctx, ancestor_sets
        )
        all_results[method_name] = sort_and_rank(results, config.top_k)

    return all_results


def main() -> None:
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=ALL_METHODS,
        run_fn=run,
        output_dir=SEMANTIC_DIR,
    )


if __name__ == "__main__":
    main()
