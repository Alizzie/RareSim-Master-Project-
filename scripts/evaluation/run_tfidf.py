"""
RareSim TF-IDF Batch Runner

Runs the TF-IDF similarity method on every test case and caches results.

Methods:
    tfidf

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage:
    python evaluation/run_tfidf.py \\
        --test-set test_data/test_cases/MME.json
"""

import argparse
import time
from pathlib import Path

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    TFIDF_METHODS,
    load_test_cases,
    build_patient,
    serialize_results,
    cache_path_for,
    methods_already_cached,
    save_cache,
    print_header,
    print_case,
    print_case_ok,
    print_case_err,
    print_summary,
    add_common_args,
)

from raresim.core.context import AppContext
from raresim.utils.math import preprocess_ancestor_sets
from raresim.core.pipeline import PipelineConfig
from raresim.types.schemas import PatientProfile
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run TF-IDF on every test case."""

    if config is None:
        config = PipelineConfig()

    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("tfidf", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    print("Loading shared context...")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, config.use_canonical_profiles)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print(f"  HPO labels       : {ctx.app_metadata.n_hpo_labels}")
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print("  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, TFIDF_METHODS):
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print_case(index, total, hpo_terms, ground_truth)

        try:
            t0 = time.time()
            results = run_tfidf(patient, TFIDF_METHODS, config, ctx)
            elapsed = time.time() - t0
            total_time += elapsed

            save_cache(
                cache_file,
                index,
                hpo_terms,
                ground_truth,
                serialize_results(results),
                {"tfidf": round(elapsed, 3)},
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
        description="RareSim TF-IDF batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        top_k=args.top_k,
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
