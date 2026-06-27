"""
TF-IDF similarity pipeline — three modes.

tfidf_hpo    : HPO term sets weighted by IDF over the disease HPO corpus.
               Binary TF. Patient and disease documents are HPO ID sets.

tfidf_text   : Clinical word tokens weighted by IDF over disease descriptions.
               Raw TF (word counts). Patient document is raw_text.
               Disease document is merged_description.

tfidf_hybrid : Patient HPO label tokens vs disease description text.
               Bridges HPO and free-text without requiring HPO extraction
               on the disease side.

All three modes use the same cosine similarity scorer and produce
MethodResults with the same explanation schema.
"""

from raresim.core.context import AppContext
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.tfidf.config import (
    ALL_METHODS,
    DISEASE_TEXT_FIELD,
    METHOD_HPO,
    METHOD_HYBRID,
    METHOD_TEXT,
    METHOD_HPO_LABELS,
    PIPELINE_NAME,
    TFIDF_DIR,
)
from raresim.similarity_methods.tfidf.explanation import build_explanation
from raresim.similarity_methods.tfidf.methods import (
    build_disease_text_vector,
    build_hpo_label_vector,
    build_patient_text_vector,
    build_tfidf_vector,
    compute_idf,
    compute_text_idf,
)
from raresim.types.result import MethodResults, RunStats, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.hpo_utils import filter_terms_by_ic
from raresim.utils.similarity_math import cosine_similarity
from raresim.utils.timer import Timer


def _build_result(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    disease_id: str,
    profile: dict,
    score: float,
    method_name: str,
    explanation: dict,
    ctx: AppContext,
) -> SimilarityResult:
    """Build a SimilarityResult with category/profile metadata attached."""
    category_metadata = build_category_metadata(
        disease_id=disease_id,
        profile=profile,
        disease_ancestors=ctx.disease_ancestors,
        disease_metadata_index=ctx.disease_metadata_index,
    )

    return SimilarityResult(
        disease_id=disease_id,
        label=profile.get("label", ""),
        profile_type=category_metadata["profile_type"],
        category_source_id=category_metadata["category_source_id"],
        category_path=category_metadata["category_path"],
        matched_aliases=category_metadata["matched_aliases"],
        score=score,
        method_name=method_name,
        explanation=explanation,
    )


def _empty_stats(patient: PatientProfile) -> RunStats:
    """Return empty run statistics for skipped TF-IDF modes."""
    return build_run_stats(
        n_patient_terms_raw=len(patient.hpo_terms),
        n_patient_terms_propagated=len(patient.get_terms(use_propagated=True)),
        n_patient_terms_used=0,
        n_diseases_scored=0,
        n_diseases_skipped=0,
        computation_time=0.0,
    )


# ── HPO mode ──────────────────────────────────────────────────────────────────


def _run_hpo_mode(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    config: PipelineConfig,
    ctx: AppContext,
) -> tuple[list[SimilarityResult], RunStats]:
    """
    Run the HPO-based TF-IDF similarity method.

    Patient and disease documents are sets of HPO IDs.
    Binary TF. IDF carries the weighting signal.
    """
    patient_raw_terms = set(patient.hpo_terms)
    patient_terms = set(patient.get_terms(config.use_propagated_terms))

    if not patient_terms:
        print(f"[{METHOD_HPO}] Patient has no HPO terms — skipping.")
        return [], _empty_stats(patient)

    idf = compute_idf(ctx.disease_profiles, propagated_term_key=config.terms_key)
    patient_vec = build_tfidf_vector(patient_terms, idf)

    if not patient_vec:
        print(
            f"[{METHOD_HPO}] Patient HPO terms produced no "
            "IDF-weighted vector — skipping."
        )
        return [], _empty_stats(patient)

    method_timer = Timer(METHOD_HPO).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))

        if not disease_terms:
            n_skipped += 1
            continue

        disease_vec = build_tfidf_vector(disease_terms, idf)
        score = cosine_similarity(patient_vec, disease_vec)

        explanation = build_explanation(
            tfidf_mode=METHOD_HPO,
            patient_terms=patient_terms,
            disease_terms=disease_terms,
            patient_vec=patient_vec,
            disease_vec=disease_vec,
            idf=idf,
            score=score,
            hpo_labels=ctx.hpo_labels,
            ic_values=ctx.ic_values,
            patient_raw_terms=patient_raw_terms,
        )

        results.append(
            _build_result(
                disease_id=disease_id,
                profile=profile,
                score=score,
                method_name=METHOD_HPO,
                explanation=explanation.to_dict(),
                ctx=ctx,
            )
        )

    stats = build_run_stats(
        n_patient_terms_raw=len(patient_raw_terms),
        n_patient_terms_propagated=len(patient.get_terms(use_propagated=True)),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=n_skipped,
        computation_time=method_timer.stop(),
    )

    return results, stats


# ── Text mode ─────────────────────────────────────────────────────────────────


def _run_text_mode(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    config: PipelineConfig,
    ctx: AppContext,
) -> tuple[list[SimilarityResult], RunStats]:
    """Run the clinical-text TF-IDF similarity method."""
    patient_raw_text = patient.raw_text or ""

    if not patient_raw_text.strip():
        print(f"[{METHOD_TEXT}] No raw_text on patient — skipping.")
        return [], _empty_stats(patient)

    idf = compute_text_idf(ctx.disease_profiles, text_field=DISEASE_TEXT_FIELD)
    patient_vec = build_patient_text_vector(patient_raw_text, idf)

    if not patient_vec:
        print(f"[{METHOD_TEXT}] Patient text produced no tokens — skipping.")
        return [], _empty_stats(patient)

    method_timer = Timer(METHOD_TEXT).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_vec, used_fallback = build_disease_text_vector(
            profile,
            idf,
            DISEASE_TEXT_FIELD,
        )

        if not disease_vec:
            n_skipped += 1
            continue

        score = cosine_similarity(patient_vec, disease_vec)

        explanation = build_explanation(
            tfidf_mode=METHOD_TEXT,
            patient_vec=patient_vec,
            disease_vec=disease_vec,
            idf=idf,
            score=score,
            hpo_labels=ctx.hpo_labels,
            ic_values=ctx.ic_values,
            patient_terms=set(),
            disease_terms=set(),
            patient_raw_terms=None,
            extra_diagnostics={"disease_used_label_fallback": used_fallback},
        )

        results.append(
            _build_result(
                disease_id=disease_id,
                profile=profile,
                score=score,
                method_name=METHOD_TEXT,
                explanation=explanation.to_dict(),
                ctx=ctx,
            )
        )

    stats = build_run_stats(
        n_patient_terms_raw=len(patient.hpo_terms),
        n_patient_terms_propagated=len(patient.get_terms(use_propagated=True)),
        n_patient_terms_used=len(patient_vec),
        n_diseases_scored=len(results),
        n_diseases_skipped=n_skipped,
        computation_time=method_timer.stop(),
    )

    return results, stats


# ── Hybrid mode ───────────────────────────────────────────────────────────────


def _run_hybrid_mode(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    config: PipelineConfig,
    ctx: AppContext,
) -> tuple[list[SimilarityResult], RunStats]:
    """Run the hybrid HPO-label-to-disease-text TF-IDF method."""
    all_patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_raw_terms = set(patient.hpo_terms)
    patient_terms = filter_terms_by_ic(
        all_patient_terms,
        ctx.ic_values,
        config.ic_threshold,
    )

    idf = compute_text_idf(ctx.disease_profiles, text_field=DISEASE_TEXT_FIELD)
    patient_vec = build_hpo_label_vector(patient_terms, ctx.hpo_labels, idf)

    if not patient_vec:
        print(f"[{METHOD_HYBRID}] Patient HPO labels produced no tokens — skipping.")
        return [], _empty_stats(patient)

    method_timer = Timer(METHOD_HYBRID).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_vec, used_fallback = build_disease_text_vector(
            profile,
            idf,
            DISEASE_TEXT_FIELD,
        )

        if not disease_vec:
            n_skipped += 1
            continue

        score = cosine_similarity(patient_vec, disease_vec)

        explanation = build_explanation(
            tfidf_mode=METHOD_HYBRID,
            patient_vec=patient_vec,
            disease_vec=disease_vec,
            idf=idf,
            score=score,
            hpo_labels=ctx.hpo_labels,
            ic_values=ctx.ic_values,
            patient_terms=patient_terms,
            disease_terms=set(),
            patient_raw_terms=patient_raw_terms,
            all_patient_terms_before_filter=all_patient_terms,
            extra_diagnostics={"disease_used_label_fallback": used_fallback},
        )

        results.append(
            _build_result(
                disease_id=disease_id,
                profile=profile,
                score=score,
                method_name=METHOD_HYBRID,
                explanation=explanation.to_dict(),
                ctx=ctx,
            )
        )

    stats = build_run_stats(
        n_patient_terms_raw=len(patient_raw_terms),
        n_patient_terms_propagated=len(all_patient_terms),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=n_skipped,
        computation_time=method_timer.stop(),
    )

    return results, stats


# ── HPO Labels mode ───────────────────────────────────────────────────────────────
def _run_hpo_labels_mode(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    config: PipelineConfig,
    ctx: AppContext,
) -> tuple[list[SimilarityResult], RunStats]:
    """Run the HPO-label-to-HPO-label TF-IDF method."""
    all_patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_raw_terms = set(patient.hpo_terms)
    patient_terms = filter_terms_by_ic(
        all_patient_terms,
        ctx.ic_values,
        config.ic_threshold,
    )

    idf = compute_text_idf(ctx.disease_profiles, DISEASE_TEXT_FIELD)
    patient_vec = build_hpo_label_vector(patient_terms, ctx.hpo_labels, idf)

    if not patient_vec:
        print(
            f"[{METHOD_HPO_LABELS}] Patient HPO labels produced no tokens — skipping."
        )
        return [], _empty_stats(patient)

    method_timer = Timer(METHOD_HPO_LABELS).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))

        if not disease_terms:
            n_skipped += 1
            continue

        disease_vec = build_hpo_label_vector(disease_terms, ctx.hpo_labels, idf)
        score = cosine_similarity(patient_vec, disease_vec)

        explanation = build_explanation(
            tfidf_mode=METHOD_HPO_LABELS,
            patient_vec=patient_vec,
            disease_vec=disease_vec,
            idf=idf,
            score=score,
            hpo_labels=ctx.hpo_labels,
            ic_values=ctx.ic_values,
            patient_terms=patient_terms,
            disease_terms=disease_terms,
            patient_raw_terms=patient_raw_terms,
            all_patient_terms_before_filter=all_patient_terms,
        )

        results.append(
            _build_result(
                disease_id=disease_id,
                profile=profile,
                score=score,
                method_name=METHOD_HPO_LABELS,
                explanation=explanation.to_dict(),
                ctx=ctx,
            )
        )

    stats = build_run_stats(
        n_patient_terms_raw=len(patient_raw_terms),
        n_patient_terms_propagated=len(all_patient_terms),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=n_skipped,
        computation_time=method_timer.stop(),
    )

    return results, stats


_MODE_RUNNERS = {
    METHOD_HPO: _run_hpo_mode,
    METHOD_TEXT: _run_text_mode,
    METHOD_HYBRID: _run_hybrid_mode,
    METHOD_HPO_LABELS: _run_hpo_labels_mode,
}


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the TF-IDF similarity pipeline for selected methods."""
    all_results: dict[str, MethodResults] = {}

    for method_name in ALL_METHODS:
        if method_name not in selected:
            continue

        runner = _MODE_RUNNERS[method_name]
        results, stats = runner(patient, config, ctx)

        if results:
            all_results[method_name] = sort_and_rank(
                results,
                config,
                stats,
                method_name,
                PIPELINE_NAME,
            )

    return all_results


def main() -> None:
    """Run the TF-IDF similarity pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=ALL_METHODS,
        run_fn=run,
        output_dir=TFIDF_DIR,
    )


if __name__ == "__main__":
    main()
