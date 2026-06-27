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

import hashlib
from pathlib import Path
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
    embed_term_set,
    generate_walks,
    train_word2vec,
)
from raresim.similarity_methods.hpo2vec.config import (
    HPO2VEC_DIR,
    MODEL_CACHE_DIR,
    ALL_METHOD,
    PIPELINE_NAME,
    WALK_LENGTH,
    WALKS_PER_NODE,
    P,
    Q,
    EMBEDDING_DIM,
    WINDOW_SIZE,
    MIN_COUNT,
    EPOCHS,
)
from raresim.similarity_methods.hpo2vec.explanation import build_explanation
from raresim.types.result import MethodResults, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.timer import Timer
from raresim.utils.similarity_math import cosine_similarity_dense


def _model_cache_path(terms_key: str) -> Path:
    """
    Build a cache path keyed on every parameter that changes the trained model.

    Any change to walk params, embedding params, or which term set feeds the
    graph produces a different filename, so a parameter sweep never silently
    loads a model trained under different settings.
    """
    key_params = {
        "walk_length": WALK_LENGTH,
        "walks_per_node": WALKS_PER_NODE,
        "p": P,
        "q": Q,
        "embedding_dim": EMBEDDING_DIM,
        "window_size": WINDOW_SIZE,
        "min_count": MIN_COUNT,
        "epochs": EPOCHS,
        "terms_key": terms_key,
    }
    key = "_".join(f"{name}={value}" for name, value in sorted(key_params.items()))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return MODEL_CACHE_DIR / f"hpo2vec_{digest}.model"


def load_or_train(
    disease_profiles: dict[str, dict],
    ic_values: dict[str, float],
    hpo_parents: dict[str, list[str]],
    terms_key: str = "hpo_terms",
) -> Word2Vec:
    """Load a saved HPO2Vec model or train a new one if not found."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _model_cache_path(terms_key)

    if model_path.exists():
        print("  Loading saved HPO2Vec model...")
        return Word2Vec.load(str(model_path))

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

    model.save(str(model_path))
    print(f"  Model saved to: {model_path}")

    return model


def run(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the HPO2Vec+ similarity pipeline."""

    all_results: dict[str, MethodResults] = {}

    for method_name in selected:
        if method_name not in ALL_METHOD:
            continue

        model = load_or_train(
            disease_profiles=ctx.disease_profiles,
            ic_values=ctx.ic_values,
            hpo_parents=ctx.hpo_parents,
            terms_key="hpo_terms",
        )

        patient_raw_terms = patient.hpo_terms
        patient_terms = patient.get_terms(config.use_propagated_terms)
        patient_vec = embed_term_set(patient_terms, model, ctx.ic_values)
        n_terms_in_vocab = sum(1 for term in patient_terms if term in model.wv)

        if patient_vec is None:
            print(
                "[hpo2vec] No patient terms found in Word2Vec vocabulary. "
                "Check shared artifacts or retrain the HPO2Vec model."
            )
            return {}

        method_timer = Timer(method_name).start()
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

            score = cosine_similarity_dense(patient_vec, disease_vec)

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
                    explanation=build_explanation(
                        method_name=method_name,
                        score=score,
                        patient_terms=patient_terms,
                        disease_terms=disease_terms,
                        hpo_labels=ctx.hpo_labels,
                        ic_values=ctx.ic_values,
                        patient_raw_terms=patient_raw_terms,
                        n_terms_in_vocab=n_terms_in_vocab,
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

        all_results[method_name] = sort_and_rank(
            results,
            config,
            stats,
            method_name,
            PIPELINE_NAME,
        )
    return all_results


def main() -> None:
    """Load shared artifacts and run the HPO2Vec+ pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=ALL_METHOD,
        run_fn=run,
        output_dir=HPO2VEC_DIR,
    )


if __name__ == "__main__":
    main()
