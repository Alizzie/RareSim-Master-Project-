"""
Pipeline:
  1. Build a graph from HPO parents and disease profiles
  2. Run IC weighted random walks from every node
  3. Train Word2Vec on the walks
  4. For patient: aggregate their HPO term embeddings into 1 patient vector
  5. For each disease: same as 4 but for HPO
  6. Rank diseases by some similarity to the patient vector
"""

from gensim.models import Word2Vec

from raresim.types.schemas import PatientProfile
from raresim.core.context import AppContext
from raresim.utils.paths import OUTPUTS_DIR, HPO_PARENTS_PATH
from raresim.types.result import SimilarityResult,  MethodResults
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.io import load_json
from raresim.utils.timer import Timer

from raresim.similarity_methods.hpo2vec.methods import (
    build_graph,
    generate_walks,
    train_word2vec,
    embed_term_set,
    cosine_similarity_np,
)

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
    HPO2VEC_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print("  Loading saved HPO2Vec model...")
        return Word2Vec.load(str(MODEL_PATH))

    print("  No saved model found, training from scratch...")

    print("  Building graph...")
    graph = build_graph(hpo_parents, disease_profiles, terms_key=terms_key)
    print(f"  Nodes: {len(graph)}")

    print("  Generating random walks...")
    walks = generate_walks(graph, ic_values)

    print("  Training Word2Vec...")
    model = train_word2vec(walks)

    model.save(str(MODEL_PATH))
    print(f"  Model saved to: {MODEL_PATH}")

    return model


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the HPO2Vec+ similarity pipeline."""

    if METHOD_NAME not in selected:
        return {}

    hpo_parents = load_json(HPO_PARENTS_PATH)

    # Load or train model
    model = load_or_train(
        disease_profiles=ctx.disease_profiles,
        ic_values=ctx.ic_values,
        hpo_parents=hpo_parents,
        terms_key="hpo_terms",  # use raw terms for graph, propagated for embedding
    )

    # Embed the patient
    patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_vec = embed_term_set(patient_terms, model, ctx.ic_values)

    if patient_vec is None:
        print(
            "No patient terms found in Word2Vec vocabulary — check your shared artifacts."
        )
        return {}

    timer = Timer(METHOD_NAME).start()
    results = []

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))
        if not disease_terms:
            continue

        disease_vec = embed_term_set(disease_terms, model, ctx.ic_values)
        if disease_vec is None:
            continue

        score = cosine_similarity_np(patient_vec, disease_vec)

        matching_terms = sorted(patient_terms & disease_terms)

        results.append(
            SimilarityResult(
                disease_id=disease_id,
                label=profile.get("label", ""),
                score=score,
                method_name=METHOD_NAME,
                explanation={
                    "method": METHOD_NAME,
                    "score": score,
                    "matching_terms": matching_terms,
                    "n_matching": len(matching_terms),
                    "top_ic_matches": sorted(
                        [
                            {"term": t, "ic": ctx.ic_values.get(t, 0.0)}
                            for t in matching_terms
                        ],
                        key=lambda x: x["ic"],
                        reverse=True,
                    )[:5],
                },
            )
        )

    metadata = build_run_stats(
    n_patient_terms_raw=len(patient_terms),
    n_patient_terms_propagated=len(patient_terms),
    n_patient_terms_used=len(patient_terms),
    n_diseases_scored=len(results),
    n_diseases_skipped=0,
    computation_time=timer.stop(),
)

    return {METHOD_NAME: sort_and_rank(results, config, metadata, METHOD_NAME, PIPELINE_NAME)}


def main() -> None:
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=[METHOD_NAME],
        run_fn=run,
        output_dir=HPO2VEC_DIR,
    )


if __name__ == "__main__":
    main()
