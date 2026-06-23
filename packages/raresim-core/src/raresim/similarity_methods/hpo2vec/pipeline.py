"""
HPO2Vec+ similarity pipeline.

Pipeline:
  1. Build a graph from HPO parents and disease profiles
  2. Run IC weighted random walks from every node
  3. Train Word2Vec on the walks
  4. For patient: aggregate their HPO term embeddings into 1 patient vector
  5. For each disease: same as 4 but for HPO
  6. Rank diseases by some similarity to the patient vector
"""

from gensim.models import Word2Vec

from raresim.core.context import AppContext
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.hpo2vec.methods import (
    build_graph,
    cosine_similarity_np,
    embed_term_set,
    generate_walks,
    train_word2vec,
)
from raresim.types.result import MethodResults, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.io import load_json
from raresim.utils.paths import HPO_PARENTS_PATH, OUTPUTS_DIR
from raresim.utils.timer import Timer

HPO2VEC_DIR = OUTPUTS_DIR / "hpo2vec"
PIPELINE_NAME = "hpo2vec"
METHOD_NAME = "hpo2vec_plus"

MODEL_PATH = HPO2VEC_DIR / "hpo2vec_model"


def load_or_train(
    disease_profiles: dict[str, dict],
    ic_values: dict[str, float],
    hpo_parents: dict[str, list[str]],
    terms_key: str = "hpo_terms",
) -> Word2Vec:
    """Load a saved HPO2Vec model or train a new one if not found."""
    HPO2VEC_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("  Loading saved HPO2Vec model...")
        return Word2Vec.load(str(MODEL_PATH))

    print("  No saved model found, training from scratch...")

    print("  Building graph...")
    graph = build_graph(
        hpo_parents,
        disease_profiles,
        terms_key=terms_key,
    )
    print(f"  Nodes: {len(graph)}")

    print("  Generating random walks...")
    walks = generate_walks(graph, ic_values)

    print("  Training Word2Vec...")
    model = train_word2vec(walks)

    model.save(str(MODEL_PATH))
    print(f"  Model saved to: {MODEL_PATH}")

    return model


def _build_explanation(
    method_name: str,
    score: float,
    matching_terms: list[str],
    ic_values: dict[str, float],
) -> dict:
    """Build a simple explanation for HPO2Vec results."""
    return {
        "method": method_name,
        "score": score,
        "matching_terms": matching_terms,
        "n_matching": len(matching_terms),
        "top_ic_matches": sorted(
            [
                {
                    "term": term,
                    "ic": ic_values.get(term, 0.0),
                }
                for term in matching_terms
            ],
            key=lambda item: item["ic"],
            reverse=True,
        )[:5],
    }


def run(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the HPO2Vec+ similarity pipeline."""
    if METHOD_NAME not in selected:
        return {}

    hpo_parents = load_json(HPO_PARENTS_PATH)

    model = load_or_train(
        disease_profiles=ctx.disease_profiles,
        ic_values=ctx.ic_values,
        hpo_parents=hpo_parents,
        terms_key="hpo_terms",
    )

    patient_raw_terms = set(patient.hpo_terms)
    patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_vec = embed_term_set(patient_terms, model, ctx.ic_values)

    if patient_vec is None:
        print(
            "[hpo2vec] No patient terms found in Word2Vec vocabulary. "
            "Check shared artifacts or retrain the HPO2Vec model."
        )
        return {}

    method_timer = Timer(METHOD_NAME).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))

        if not disease_terms:
            n_skipped += 1
            continue

        disease_vec = embed_term_set(disease_terms, model, ctx.ic_values)

        if disease_vec is None:
            n_skipped += 1
            continue

        score = cosine_similarity_np(patient_vec, disease_vec)
        matching_terms = sorted(patient_terms & disease_terms)

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
                method_name=METHOD_NAME,
                explanation=_build_explanation(
                    method_name=METHOD_NAME,
                    score=score,
                    matching_terms=matching_terms,
                    ic_values=ctx.ic_values,
                ),
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

    return {
        METHOD_NAME: sort_and_rank(
            results,
            config,
            stats,
            METHOD_NAME,
            PIPELINE_NAME,
        )
    }


def main() -> None:
    """Load shared artifacts and run the HPO2Vec+ pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=[METHOD_NAME],
        run_fn=run,
        output_dir=HPO2VEC_DIR,
    )


if __name__ == "__main__":
    main()
