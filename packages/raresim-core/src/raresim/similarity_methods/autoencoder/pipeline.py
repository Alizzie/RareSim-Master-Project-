"""
Denoising Autoencoder pipeline

Pipeline:
  1. Build vocabulary from all HPO terms in disease profiles
  2. Convert each disease profile to a binary HPO vector
  3. Train a denoising autoencoder on those vectors (or load saved model)
  4. Use the trained encoder to embed both patient and diseases into latent space
  5. Rank diseases by cosine similarity to the patient's latent vector

The denoising part:
  - During training, HPO terms are randomly dropped from the input (noise)
  - The model learns to reconstruct the full clean vector from the corrupted one
  - This forces the encoder to learn good latent representations that capture
    the underlying phenotype structure, not just memorize the term presence
"""

import hashlib
from pathlib import Path
import numpy as np

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig, build_run_stats, sort_and_rank
from raresim.ontology.disease_category import build_category_metadata
from raresim.types.result import SimilarityResult, MethodResults
from raresim.similarity_methods.autoencoder.methods import (
    DenoisingAutoencoder,
    build_vocabulary,
    terms_to_vector,
    euclidean_similarity,
)
from raresim.similarity_methods.autoencoder.explanation import build_explanation
from raresim.types.schemas import PatientProfile
from raresim.utils._pipeline_runner import run_pipeline_main
from raresim.utils.io import save_json, load_json
from raresim.utils.timer import Timer


from raresim.similarity_methods.autoencoder.config import (
    ALL_METHOD,
    METHOD_NAME,
    PIPELINE_NAME,
    AUTOENCODER_DIR,
    MODEL_CACHE_DIR,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MOMENTUM,
    NOISE_RATE,
    EPOCHS,
    BATCH_SIZE,
)


def _cache_paths(disease_profiles: dict, terms_key: str) -> tuple[Path, Path]:
    """
    Build model + vocab cache paths keyed on the inputs that change the model.

    Includes the vocabulary fingerprint (the model's input dimension depends on
    it) plus terms_key and the architecture/training hyperparameters, so a
    different profile set or param sweep never loads a mismatched model.
    """
    vocab = build_vocabulary(disease_profiles, terms_key)
    key_params = {
        "n_vocab": len(vocab),
        "vocab_hash": hashlib.sha256("|".join(vocab).encode("utf-8")).hexdigest()[:12],
        "terms_key": terms_key,
        "hidden": HIDDEN_DIM,
        "latent": LATENT_DIM,
        "lr": LEARNING_RATE,
        "momentum": MOMENTUM,
        "noise": NOISE_RATE,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
    }
    key = "_".join(f"{k}={v}" for k, v in sorted(key_params.items()))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return (
        MODEL_CACHE_DIR / f"autoencoder_{digest}.npz",
        MODEL_CACHE_DIR / f"vocab_{digest}.json",
    )


def _train(
    disease_profiles: dict[str, dict],
    vocab: list[str],
    term_to_idx: dict[str, int],
    terms_key: str = "propagated_hpo_terms",
) -> DenoisingAutoencoder:
    """
    Build vocabulary, vectorize disease profiles, and train the autoencoder.

    Returns:
        trained model, vocabulary list, and term-to-index mapping.
    """

    print("  Vectorizing disease profiles...")
    vectors = np.array(
        [
            terms_to_vector(
                set(profile.get(terms_key, [])),
                vocab,
                term_to_idx,
            )
            for profile in disease_profiles.values()
        ],
        dtype=np.float32,
    )

    print(f"  Training matrix shape: {vectors.shape}")
    print("  Training denoising autoencoder...")

    model = DenoisingAutoencoder(
        vocab_size=len(vocab),
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        noise_rate=NOISE_RATE,
    )
    model.train(vectors, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return model


def load_or_train(
    disease_profiles: dict[str, dict],
    terms_key: str = "propagated_hpo_terms",
) -> tuple[DenoisingAutoencoder, list[str], dict[str, int]]:
    """Load a cached model for this profile-set + param combination, or train."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path, vocab_path = _cache_paths(disease_profiles, terms_key)

    if model_path.exists() and vocab_path.exists():
        print(f"  Loading cached autoencoder: {model_path.name}")
        model = DenoisingAutoencoder.load(model_path)
        vocab = [str(t) for t in load_json(vocab_path)]
        return model, vocab, {t: i for i, t in enumerate(vocab)}

    print("  No cached model for these inputs, training from scratch...")
    vocab = [str(t) for t in build_vocabulary(disease_profiles, terms_key)]
    term_to_idx = {t: i for i, t in enumerate(vocab)}
    print(f"  Vocabulary size: {len(vocab)} HPO terms")
    model = _train(disease_profiles, vocab, term_to_idx, terms_key)

    model.save(model_path)
    save_json(vocab, vocab_path)
    print(f"  Model saved to: {model_path}")

    return model, vocab, term_to_idx


def run(  # pylint: disable=too-many-locals
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, MethodResults]:
    """Run the denoising autoencoder similarity pipeline."""
    if METHOD_NAME not in selected:
        return {}

    model, vocab, term_to_idx = load_or_train(
        ctx.disease_profiles,
        terms_key=config.terms_key,
    )

    patient_raw_terms = patient.hpo_terms
    patient_terms = patient.get_terms(config.use_propagated_terms)

    if not patient_terms:
        print("[autoencoder] Warning: patient has no HPO terms.")
        return {}

    patient_vec = terms_to_vector(patient_terms, vocab, term_to_idx)
    patient_latent = model.encode(patient_vec.reshape(1, -1))[0]

    timer = Timer(METHOD_NAME).start()
    results = []
    n_skipped = 0

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))

        if not disease_terms:
            n_skipped += 1
            continue

        disease_vec = terms_to_vector(disease_terms, vocab, term_to_idx)
        disease_latent = model.encode(disease_vec.reshape(1, -1))[0]
        score = euclidean_similarity(patient_latent, disease_latent)

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
                explanation=build_explanation(
                    method_name=METHOD_NAME,
                    score=score,
                    patient_terms=patient_terms,
                    disease_terms=disease_terms,
                    hpo_labels=ctx.hpo_labels,
                    ic_values=ctx.ic_values,
                    patient_raw_terms=patient_raw_terms,
                ),
            )
        )

    metadata = build_run_stats(
        n_patient_terms_raw=len(patient_raw_terms),
        n_patient_terms_propagated=len(patient.get_terms(use_propagated=True)),
        n_patient_terms_used=len(patient_terms),
        n_diseases_scored=len(results),
        n_diseases_skipped=n_skipped,
        computation_time=timer.stop(),
    )

    return {
        METHOD_NAME: sort_and_rank(
            results, config, metadata, METHOD_NAME, PIPELINE_NAME
        )
    }


def main() -> None:
    """Load shared artifacts and run the autoencoder pipeline."""
    run_pipeline_main(
        pipeline_name=PIPELINE_NAME,
        method_names=ALL_METHOD,
        run_fn=run,
        output_dir=AUTOENCODER_DIR,
    )


if __name__ == "__main__":
    main()
