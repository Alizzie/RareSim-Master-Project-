"""Batch runner for RareSim TF-IDF similarity.
Usage:
    python -m scripts.evaluation.run_tfidf \
        --test-set <path_to_test_set.json> \
        [--no-resume] \
        [--limit <max_cases>] \
        [--top-k <top_k_results>]
"""

# pylint: disable=broad-exception-caught

import argparse
from pathlib import Path

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    TFIDF_METHODS,
    add_common_args,
    build_patient,
    cache_path_for,
    load_test_cases,
    methods_already_cached,
    print_case,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
    serialize_results,
)


def run(  # pylint: disable=too-many-locals
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
    if limit is not None:
        cases = cases[:limit]

    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    print("Loading shared context...")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, config.use_canonical_profiles)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print(f"  HPO labels       : {ctx.app_metadata.n_hpo_labels}")
    print("  Ready.\n")

    skipped = 0
    processed = 0
    failed = 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, TFIDF_METHODS):
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print_case(index, total, hpo_terms, ground_truth)

        try:
            case_timer = Timer("tfidf").start()

            results = run_tfidf(patient, TFIDF_METHODS, config, ctx)

            elapsed = round(case_timer.stop(), 3)
            total_time += elapsed

            save_cache(
                cache_file,
                index,
                hpo_terms,
                ground_truth,
                serialize_results(results),
                {"tfidf": elapsed},
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
        description="RareSim TF-IDF batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the CLI entry point."""
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
