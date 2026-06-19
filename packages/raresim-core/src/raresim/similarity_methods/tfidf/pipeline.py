"""
TF-IDF cosine similarity pipeline.

Pipeline:
  1. Build corpus of disease profiles (build_shared_artifacts.py)
  2. Compute TF-IDF vectors for each disease
  3. Treat the patient's HPO terms as a query document and compute their TF-IDF vector
  4. Compare patient vector to each disease vector using cosine similarity
  5. Rank diseases by similarity score
"""

from raresim.types.schemas import PatientProfile
from raresim.core.context import AppContext
from raresim.utils.paths import OUTPUTS_DIR
from raresim.types.result import MethodResults, SimilarityResult
from raresim.utils.shared_methods import cosine_similarity
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.explanation import (
    expand,
    SET_BASED_EXPLANATION,
    with_top_idf_weighted_terms,
)
from raresim.similarity_methods.tfidf.methods import build_tfidf_vector, compute_idf
from raresim.utils.timer import Timer

TFIDF_DIR = OUTPUTS_DIR / "tfidf"
PIPELINE_NAME = "tfidf"
METHOD_NAME = "tfidf"


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the TF-IDF similarity pipeline for the given patient and selected methods."""

    if METHOD_NAME not in selected:
        return {}

    patient_terms = set(patient.get_terms(config.use_propagated_terms))

    # Compute IDF from the disease profiles (corpus)
    idf = compute_idf(ctx.disease_profiles, propagated_term_key=config.terms_key)
    patient_vec = build_tfidf_vector(patient_terms, idf)

    timer = Timer(METHOD_NAME).start()
    results = []
    skipped = 0
    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))
        if not disease_terms:
            skipped += 1
            continue

        disease_vec = build_tfidf_vector(disease_terms, idf)
        score = cosine_similarity(patient_vec, disease_vec)

        results.append(
            SimilarityResult(
                disease_id=disease_id,
                label=profile.get("label", ""),
                score=score,
                method_name=METHOD_NAME,
                explanation=expand(
                    {},
                    patient_terms,
                    disease_terms,
                    expanders=[
                        *SET_BASED_EXPLANATION,
                        with_top_idf_weighted_terms(idf, top_n=5),
                    ],
                ),
            )
        )

    metadata = build_run_stats(
        n_patient_terms_raw=len(patient_terms),
        n_patient_terms_propagated=len(patient_terms),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=skipped,
        computation_time=timer.stop(),
    )

    return {"tfidf_cosine": sort_and_rank(
        results,
        config,
        metadata,
        method_name=METHOD_NAME,
        pipeline_name=PIPELINE_NAME,
    )}


def main() -> None:
    """Example main function to run the tfidf similarity pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=[METHOD_NAME],
        run_fn=run,
        output_dir=TFIDF_DIR,
    )


if __name__ == "__main__":
    main()
