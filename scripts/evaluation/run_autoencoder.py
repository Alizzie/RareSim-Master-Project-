"""Batch runner for RareSim denoising autoencoder similarity.
Usage:
    python -m scripts.evaluation.run_autoencoder \
        --test-set <path_to_test_set.json> \
        [--no-resume] \
        [--limit <max_cases>] \
        [--top-k <top_k_results>]
"""

# pylint: disable=broad-exception-caught,too-many-locals

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.autoencoder.methods import terms_to_vector
from raresim.similarity_methods.autoencoder.pipeline import AUTOENCODER_DIR
from raresim.similarity_methods.autoencoder.pipeline import METHOD_NAME
from raresim.similarity_methods.autoencoder.pipeline import load_or_train
from raresim.types.result import SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    add_common_args,
    cache_path_for,
    load_test_cases,
    methods_already_cached,
    print_case,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
)

METHOD_NAMES = [METHOD_NAME]


@dataclass(frozen=True)
class EncodedDiseases:
    """Precomputed disease latent vectors used for fast ranking."""

    disease_ids: list[str]
    disease_labels: list[str]
    disease_matrix: np.ndarray


def _delete_model_cache() -> None:
    """Delete saved autoencoder artifacts so the model is retrained."""
    model_path = AUTOENCODER_DIR / "autoencoder_model.npz"
    vocab_path = AUTOENCODER_DIR / "vocab.json"

    if model_path.exists():
        model_path.unlink()
        print("  Deleted saved model; will retrain from scratch.")

    if vocab_path.exists():
        vocab_path.unlink()


def _preencode_diseases(
    ctx: AppContext,
    config: PipelineConfig,
    model: Any,
    vocab: list[str],
    term_to_idx: dict[str, int],
) -> EncodedDiseases:
    """Encode all disease profiles into normalized latent vectors."""
    disease_ids: list[str] = []
    disease_labels: list[str] = []
    disease_latents: list[np.ndarray] = []

    for disease_id, profile in ctx.disease_profiles.items():
        disease_terms = set(profile.get(config.terms_key, []))
        if not disease_terms:
            continue

        disease_vec = terms_to_vector(disease_terms, vocab, term_to_idx)
        disease_latent = model.encode(disease_vec.reshape(1, -1))[0]

        disease_ids.append(disease_id)
        disease_labels.append(profile.get("label", ""))
        disease_latents.append(disease_latent)

    disease_matrix = np.array(disease_latents, dtype=np.float32)
    disease_norms = np.linalg.norm(disease_matrix, axis=1, keepdims=True)
    disease_norms = np.clip(disease_norms, 1e-12, None)

    return EncodedDiseases(
        disease_ids=disease_ids,
        disease_labels=disease_labels,
        disease_matrix=disease_matrix / disease_norms,
    )


def _build_result(
    rank: int,
    disease_id: str,
    disease_label: str,
    score: float,
    ctx: AppContext,
) -> dict:
    """Build one serialized autoencoder result row."""
    profile = ctx.disease_profiles[disease_id]
    category_metadata = build_category_metadata(
        disease_id=disease_id,
        profile=profile,
        disease_ancestors=ctx.disease_ancestors,
        disease_metadata_index=ctx.disease_metadata_index,
    )

    result = SimilarityResult(
        disease_id=disease_id,
        label=disease_label,
        profile_type=category_metadata["profile_type"],
        category_source_id=category_metadata["category_source_id"],
        category_path=category_metadata["category_path"],
        matched_aliases=category_metadata["matched_aliases"],
        score=score,
        method_name=METHOD_NAME,
    )
    result.rank = rank
    return result.to_dict()


def run(  # pylint: disable=too-many-statements
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
    retrain: bool = False,
) -> Path:
    """Run the denoising autoencoder on every test case."""
    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("autoencoder", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit is not None:
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
        _delete_model_cache()

    print("Loading / training autoencoder model...")
    model, vocab, term_to_idx = load_or_train(
        ctx.disease_profiles,
        terms_key=config.terms_key,
    )
    print("  Model ready.\n")

    print("Pre-encoding all disease profiles into latent space...")
    encode_start = time.time()
    encoded_diseases = _preencode_diseases(
        ctx=ctx,
        config=config,
        model=model,
        vocab=vocab,
        term_to_idx=term_to_idx,
    )
    print(
        f"  Encoded {len(encoded_diseases.disease_ids)} diseases "
        f"in {time.time() - encode_start:.1f}s\n"
    )

    skipped = 0
    processed = 0
    failed = 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, METHOD_NAMES):
            skipped += 1
            continue

        print_case(index, total, hpo_terms, ground_truth)

        try:
            wall_start = time.time()
            method_timer = Timer(METHOD_NAME).start()

            patient_terms = set(hpo_terms)
            patient_vec = terms_to_vector(patient_terms, vocab, term_to_idx)
            patient_latent = model.encode(patient_vec.reshape(1, -1))[0]
            patient_norm = max(float(np.linalg.norm(patient_latent)), 1e-12)
            normalized_patient = patient_latent / patient_norm

            scores = encoded_diseases.disease_matrix @ normalized_patient
            ranked_indices = np.argsort(-scores)[:top_k]
            computation_time = method_timer.stop()

            results = [
                _build_result(
                    rank=rank,
                    disease_id=encoded_diseases.disease_ids[disease_index],
                    disease_label=encoded_diseases.disease_labels[disease_index],
                    score=float(scores[disease_index]),
                    ctx=ctx,
                )
                for rank, disease_index in enumerate(ranked_indices, start=1)
            ]

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

        except Exception as error:
            failed += 1
            print_case_err(error)
            (cache_dir / f"case_{index:04d}.error").write_text(
                f"{type(error).__name__}: {error}",
                encoding="utf-8",
            )

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    """Run the CLI entry point."""
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
