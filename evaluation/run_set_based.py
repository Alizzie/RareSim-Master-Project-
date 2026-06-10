"""
RareSim Set-Based Batch Runner

Runs set-based similarity methods on every test case and caches results.

Methods:
    set_cosine
    set_jaccard
    set_dice
    set_overlap

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage:
    python evaluation/run_set_based.py --test-set test_data/test_cases/MME.json
"""

import argparse
import sys
import time
from pathlib import Path

from _batch_utils import (
    SRC_DIR, CACHE_BASE_DIR,
    SET_BASED_METHODS,
    load_test_cases, build_patient, serialize_results,
    cache_path_for, methods_already_cached, save_cache,
    print_header, print_case, print_case_ok, print_case_err, print_summary,
    add_common_args,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.context import AppContext
from shared.math import preprocess_ancestor_sets
from shared.pipeline import PipelineConfig
from core.schemas import PatientProfile


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run set-based similarity methods on every test case."""
    from similarity_methods.set_based.pipeline import run as run_set_based

    if config is None:
        config = PipelineConfig()

    cache_dir = CACHE_BASE_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("set-based", test_set_path, cache_dir, resume, limit)

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

        if resume and methods_already_cached(cache_file, SET_BASED_METHODS):
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print_case(index, total, hpo_terms, ground_truth)

        try:
            t0 = time.time()
            results = run_set_based(patient, SET_BASED_METHODS, config, ctx)
            elapsed = time.time() - t0
            total_time += elapsed

            method_elapsed = {
                m: round(elapsed / len(SET_BASED_METHODS), 3)
                for m in SET_BASED_METHODS
            }

            save_cache(
                cache_file, index, hpo_terms, ground_truth,
                serialize_results(results), method_elapsed, elapsed,
            )
            processed += 1
            print_case_ok(elapsed, total_time, processed, total - index - 1)

        except Exception as e:
            failed += 1
            print_case_err(e)
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim set-based similarity batch runner",
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
    