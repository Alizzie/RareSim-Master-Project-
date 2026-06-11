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

import numpy as np
from raresim.utils.io import save_json, load_json
from raresim.utils.patient_loader import load_patient
from raresim.utils.paths import OUTPUTS_DIR, PATIENT_DIR

from raresim.types.schemas import PatientProfile
from raresim.core.context import AppContext
from raresim.types.result import SimilarityResult
from raresim.core.pipeline import (
    PipelineConfig,
    build_metadata,
    sort_and_rank,
)
from raresim.utils.explanation import expand, SET_BASED_EXPLANATION
from raresim.utils.timer import Timer

from raresim.similarity_methods.autoencoder.methods import (
    DenoisingAutoencoder,
    build_vocabulary,
    terms_to_vector,
    cosine_similarity_np,
)

AUTOENCODER_DIR = OUTPUTS_DIR / "autoencoder"
PIPELINE_NAME = "autoencoder"
METHOD_NAME = "denoising_autoencoder"

# Model artifacts saved after training
MODEL_PATH = AUTOENCODER_DIR / "autoencoder_model.npz"
VOCAB_PATH = AUTOENCODER_DIR / "vocab.json"

HIDDEN_DIM = 512  # encoder/decoder hidden layer size
LATENT_DIM = 128  # size of the compressed latent representation
LEARNING_RATE = 0.01  # SGD learning rate
MOMENTUM = 0.9  # SGD momentum
NOISE_RATE = 0.2  # fraction of present HPO terms to randomly drop during training
EPOCHS = 50  # training epochs
BATCH_SIZE = 64  # minibatch size


def train_autoencoder(
    disease_profiles: dict[str, dict],
    terms_key: str = "propagated_hpo_terms",
) -> tuple[DenoisingAutoencoder, list[str], dict[str, int]]:
    """
    Build vocabulary, vectorize disease profiles, and train the autoencoder
    Returns the trained model, vocabulary list, and term -> index mapping
    """
    print("  Building vocabulary...")
    vocab = build_vocabulary(disease_profiles, terms_key)
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    print(f"  Vocabulary size: {len(vocab)} HPO terms")

    print("  Vectorizing disease profiles...")
    vectors = np.array(
        [
            terms_to_vector(set(profile.get(terms_key, [])), vocab, term_to_idx)
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

    return model, vocab, term_to_idx


def load_or_train(
    disease_profiles: dict[str, dict],
    terms_key: str = "propagated_hpo_terms",
) -> tuple[DenoisingAutoencoder, list[str], dict[str, int]]:
    """
    Load saved model and vocab if they exist, otherwise train from scratch
    Delete outputs/autoencoder/ to train again
    """
    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and VOCAB_PATH.exists():
        print("  Loading saved autoencoder model...")
        model = DenoisingAutoencoder.load(MODEL_PATH)
        vocab = load_json(VOCAB_PATH)
        term_to_idx = {term: i for i, term in enumerate(vocab)}
        print(f"  Vocabulary size: {len(vocab)} HPO terms")
        return model, vocab, term_to_idx

    print("  No saved model found, training from scratch...")
    model, vocab, term_to_idx = train_autoencoder(disease_profiles, terms_key)

    model.save(MODEL_PATH)
    save_json(vocab, VOCAB_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    return model, vocab, term_to_idx


def run(
    patient: PatientProfile,
    selected: list[str],
    config: PipelineConfig,
    ctx: AppContext,
) -> dict[str, list[SimilarityResult]]:
    """Run it"""

    if METHOD_NAME not in selected:
        return {}

    # Load or train model
    model, vocab, term_to_idx = load_or_train(
        ctx.disease_profiles, terms_key=config.terms_key
    )

    # Embed the patient
    patient_terms = set(patient.get_terms(config.use_propagated_terms))
    patient_vec = terms_to_vector(patient_terms, vocab, term_to_idx)
    patient_latent = model.encode(patient_vec.reshape(1, -1))[0]

    timer = Timer(METHOD_NAME).start()
    results = []

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))
        if not disease_terms:
            continue

        # Encode the disease into latent space
        disease_vec = terms_to_vector(disease_terms, vocab, term_to_idx)
        disease_latent = model.encode(disease_vec.reshape(1, -1))[0]

        score = cosine_similarity_np(patient_latent, disease_latent)

        results.append(
            SimilarityResult(
                disease_id=disease_id,
                label=profile.get("label", ""),
                score=score,
                method_name=METHOD_NAME,
                explanation=expand(
                    {"method": METHOD_NAME, "score": score},
                    patient_terms,
                    disease_terms,
                    expanders=SET_BASED_EXPLANATION,
                ),
            )
        )

    metadata = build_metadata(
        method_name=METHOD_NAME,
        pipeline_name=PIPELINE_NAME,
        config=config,
        n_patient_terms=len(patient_terms),
        n_disease_terms=0,
        computation_time=timer.stop(),
    )

    return {METHOD_NAME: sort_and_rank(results, metadata, config.top_k)}


def main() -> None:
    config = PipelineConfig()
    patient = load_patient(PATIENT_DIR / "example_patient.json")
    ctx = AppContext.load(
        patient=patient, use_canonical_profiles=config.use_canonical_profiles
    )

    results = run(patient, [METHOD_NAME], config, ctx)

    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {method: mr.to_dict() for method, mr in results.items()},
        AUTOENCODER_DIR / f"{PIPELINE_NAME}_top{config.top_k}.json",
    )

    for method_name, method_results in results.items():
        save_json(
            method_results.to_dict(),
            AUTOENCODER_DIR / f"{method_name}_top{config.top_k}.json",
        )
        print(f"\nTop results for {method_name}:")
        for r in method_results.rankings:
            print(
                f"  rank={r.rank:>2} | {r.disease_id:<15} | score={r.score:.4f} | {r.label}"
            )


if __name__ == "__main__":
    main()
