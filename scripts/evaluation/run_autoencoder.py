"""
RareSim Autoencoder Batch Runner

Runs the denoising autoencoder on every test case and adds results to existing cache files.
The model is trained once (or loaded from cache) and reused across all cases.

Model cache:
    outputs/autoencoder/autoencoder_model.npz
    outputs/autoencoder/vocab.json

Results cache:
    outputs/evaluation/{test_set_name}/cache/case_NNNN.json

Usage:
    python evaluation/run_autoencoder.py \\
        --test-set test_data/test_cases/MME.json
"""

import argparse
import time
from pathlib import Path

from _batch_utils import (
    load_test_cases,
    cache_path_for,
    methods_already_cached,
    save_cache,
    print_header,
    print_case,
    print_case_ok,
    print_case_err,
    print_summary,
    add_common_args,
    EVALUATION_DIR,
)

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.types.schemas import PatientProfile
from raresim.similarity_methods.autoencoder.pipeline import (
    load_or_train,
    METHOD_NAME,
    AUTOENCODER_DIR,
)
from raresim.similarity_methods.autoencoder.methods import terms_to_vector
from raresim.types.result import SimilarityResult
from raresim.utils.explanation import expand, SET_BASED_EXPLANATION
from raresim.utils.timer import Timer

METHOD_NAMES = ["denoising_autoencoder"]


def run(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
    retrain: bool = False,
) -> Path:
    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("autoencoder", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]

    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    dummy = PatientProfile("batch_init", "", set(), set())
    config = PipelineConfig(
        top_k=top_k,
        use_propagated_terms=True,
        ic_threshold=0.0,
        use_canonical_profiles=True,
    )
    ctx = AppContext.load(dummy, use_canonical_profiles=True)

    if retrain:
        model_path = AUTOENCODER_DIR / "autoencoder_model.npz"
        vocab_path = AUTOENCODER_DIR / "vocab.json"

        if model_path.exists():
            model_path.unlink()
            print("  Deleted saved model — will retrain from scratch.")

        if vocab_path.exists():
            vocab_path.unlink()

    print("Loading / training autoencoder model...")
    model, vocab, term_to_idx = load_or_train(
        ctx.disease_profiles,
        terms_key=config.terms_key,
    )
    print("  Model ready.\n")

    print("Pre-encoding all disease profiles into latent space...")
    t_encode_start = time.time()

    import numpy as np

    disease_ids = []
    disease_labels = []
    disease_latents = []

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))

        if not disease_terms:
            continue

        disease_vec = terms_to_vector(disease_terms, vocab, term_to_idx)
        disease_latent = model.encode(disease_vec.reshape(1, -1))[0]

        disease_ids.append(disease_id)
        disease_labels.append(profile.get("label", ""))
        disease_latents.append(disease_latent)

    disease_latents_matrix = np.array(disease_latents, dtype=np.float32)

    disease_norms = np.linalg.norm(disease_latents_matrix, axis=1, keepdims=True)
    disease_norms = np.clip(disease_norms, 1e-12, None)
    normalised_matrix = disease_latents_matrix / disease_norms

    print(
        f"  Encoded {len(disease_ids)} diseases "
        f"in {time.time() - t_encode_start:.1f}s\n"
    )

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, METHOD_NAMES):
            skipped += 1
            continue

        print_case(index, total, hpo_terms, ground_truth)

        try:
            wall_start = time.time()
            timer = Timer(METHOD_NAME).start()

            patient_terms = set(hpo_terms)
            patient_vec = terms_to_vector(patient_terms, vocab, term_to_idx)
            patient_latent = model.encode(patient_vec.reshape(1, -1))[0]

            patient_norm = np.linalg.norm(patient_latent)
            patient_norm = max(patient_norm, 1e-12)
            normalised_patient = patient_latent / patient_norm

            scores = normalised_matrix @ normalised_patient
            ranked_indices = np.argsort(-scores)[:top_k]

            computation_time = timer.stop()

            results = []

            for rank_idx, disease_idx in enumerate(ranked_indices, start=1):
                disease_id = disease_ids[disease_idx]
                profile = ctx.disease_profiles[disease_id]
                disease_terms = set(profile.get(config.terms_key, []))
                score = float(scores[disease_idx])

                result = SimilarityResult(
                    disease_id=disease_id,
                    label=disease_labels[disease_idx],
                    score=score,
                    method_name=METHOD_NAME,
                    explanation=expand(
                        {"method": METHOD_NAME, "score": score},
                        patient_terms,
                        disease_terms,
                        expanders=SET_BASED_EXPLANATION,
                    ),
                )
                result.rank = rank_idx

                results.append(
                    {
                        "disease_id": result.disease_id,
                        "label": result.label,
                        "score": float(result.score),
                        "method_name": result.method_name,
                        "rank": result.rank,
                        "explanation": str(result.explanation),
                    }
                )

            elapsed = round(time.time() - wall_start, 3)
            total_time += elapsed

            save_cache(
                cache_file,
                index,
                hpo_terms,
                ground_truth,
                {METHOD_NAME: results},
                {METHOD_NAME: round(computation_time, 3)},
                elapsed,
            )

            processed += 1
            print_case_ok(elapsed, total_time, processed, total - index - 1)

        except Exception as e:
            failed += 1
            print_case_err(e)
            (cache_dir / f"case_{index:04d}.error").write_text(
                f"{type(e).__name__}: {e}"
            )

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim autoencoder batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Delete saved model and retrain from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        limit=args.limit,
        top_k=args.top_k,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
