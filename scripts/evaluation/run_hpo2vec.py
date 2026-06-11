"""
RareSim HPO2Vec Batch Runner

Runs the HPO2Vec pipeline on every test case and caches results.
Loads the pre-trained Word2Vec model from outputs/hpo2vec/hpo2vec_model.

CPU only — fast after the model is loaded.

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage:
    python evaluation/run_hpo2vec.py \\
        --test-set test_data/test_cases/MME.json

Prerequisites:
    Train the model first if it does not exist:
        python pipelines/hpo2vec_pipeline.py
"""

import argparse
import time
from pathlib import Path

from _batch_utils import (
    load_test_cases,
    build_patient,
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
from raresim.utils.math import preprocess_ancestor_sets
from raresim.utils.paths import MODELS_DIR
from raresim.types.schemas import PatientProfile
from raresim.core.pipeline import PipelineConfig
from raresim.similarity_methods.hpo2vec.pipeline import run as run_hpo2vec
from raresim.similarity_methods.hpo2vec.pipeline import METHOD_NAME as HPO2VEC_METHODS

METHOD_NAME = "hpo2vec"
MODEL_PATH = MODELS_DIR / "hpo2vec_model"


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run HPO2Vec on every test case."""

    if config is None:
        config = PipelineConfig()

    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(METHOD_NAME, test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    print("Loading shared context...")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print("  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, [METHOD_NAME]):
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print_case(index, total, hpo_terms, ground_truth)

        try:
            t0 = time.time()
            rankings = run_hpo2vec(patient, [HPO2VEC_METHODS], config, ctx)
            elapsed = round(time.time() - t0, 3)
            total_time += elapsed

            save_cache(
                cache_file,
                index,
                hpo_terms,
                ground_truth,
                {METHOD_NAME: rankings},
                {METHOD_NAME: elapsed},
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
        description="RareSim HPO2Vec batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        top_k=args.top_k,
        ic_threshold=args.ic_threshold,
        use_propagated_terms=True,
        use_canonical_profiles=True,
    )

    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        config=config,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
