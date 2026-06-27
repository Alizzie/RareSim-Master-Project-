"""Batch runner for RareSim set-based similarity methods.
Usage:
    python -m scripts.evaluation.run_set_based \
        --test-set <path_to_test_set.json> \
        [--no-resume] \
        [--limit <max_cases>] \
        [--top-k <top_k_results>]
"""

# pylint: disable=broad-exception-caught,too-few-public-methods

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raresim.core.context import AppContext
from raresim.core.pipeline import PipelineConfig
from raresim.similarity_methods.set_based.pipeline import run as run_set_based
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
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

SET_BASED_METHODS = [
    "set_cosine",
    "set_jaccard",
    "set_dice",
    "set_overlap",
]


@dataclass(frozen=True)
class SharedResources:
    """Shared context and preprocessed ontology data."""

    ctx: AppContext
    ancestor_sets: dict[str, Any]


@dataclass(frozen=True)
class BatchConfig:
    """Static batch-run configuration."""

    cache_dir: Path
    resume: bool
    total_cases: int


@dataclass
class RunStats:
    """Mutable batch-run counters."""

    skipped: int = 0
    processed: int = 0
    failed: int = 0
    total_time: float = 0.0


@dataclass(frozen=True)
class CaseInput:
    """One evaluation case."""

    index: int
    hpo_terms: list[str]
    ground_truth: list[str]


@dataclass(frozen=True)
class RunnerState:
    """State required to run one set-based evaluation case."""

    config: PipelineConfig
    resources: SharedResources
    batch: BatchConfig


def _load_resources(config: PipelineConfig) -> SharedResources:
    """Load shared RareSim context and precompute HPO ancestor sets."""
    print("Loading shared context...")

    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(
        dummy,
        use_canonical_profiles=config.use_canonical_profiles,
    )
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)

    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print(f"  HPO labels       : {ctx.app_metadata.n_hpo_labels}")
    print("  Ready.\n")

    return SharedResources(ctx=ctx, ancestor_sets=ancestor_sets)


def _write_error(cache_dir: Path, case_index: int, error: Exception) -> None:
    """Write one case error file."""
    error_path = cache_dir / f"case_{case_index:04d}.error"
    error_path.write_text(
        f"{type(error).__name__}: {error}",
        encoding="utf-8",
    )


def _run_case(
    case: CaseInput,
    state: RunnerState,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], float]:
    """Run set-based methods one by one and time each method separately."""
    patient = build_patient(
        case.index,
        case.hpo_terms,
        state.resources.ancestor_sets,
    )

    all_results: dict[str, list[dict[str, Any]]] = {}
    method_elapsed: dict[str, float] = {}

    case_timer = Timer("set-based").start()

    for method in SET_BASED_METHODS:
        method_timer = Timer(method).start()

        method_results = run_set_based(
            patient,
            [method],
            state.config,
            state.resources.ctx,
        )

        method_elapsed[method] = round(method_timer.stop(), 3)
        all_results.update(serialize_results(method_results))

    elapsed = round(case_timer.stop(), 3)
    return all_results, method_elapsed, elapsed


def _handle_case(
    case: CaseInput,
    state: RunnerState,
    stats: RunStats,
) -> None:
    """Run, cache, and log one evaluation case."""
    cache_file = cache_path_for(state.batch.cache_dir, case.index)

    if state.batch.resume and methods_already_cached(cache_file, SET_BASED_METHODS):
        stats.skipped += 1
        return

    print_case(
        case.index,
        state.batch.total_cases,
        case.hpo_terms,
        case.ground_truth,
    )

    try:
        serialized_results, method_elapsed, elapsed = _run_case(case, state)
        stats.total_time += elapsed

        save_cache(
            cache_file,
            case.index,
            case.hpo_terms,
            case.ground_truth,
            serialized_results,
            method_elapsed,
            elapsed,
        )

        stats.processed += 1
        remaining = state.batch.total_cases - case.index - 1
        print_case_ok(elapsed, stats.total_time, stats.processed, remaining)

    except Exception as error:
        stats.failed += 1
        print_case_err(error)
        _write_error(state.batch.cache_dir, case.index, error)


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run set-based similarity methods on every test case."""
    if config is None:
        config = PipelineConfig(
            use_propagated_terms=True,
            use_canonical_profiles=True,
        )

    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("set-based", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    print(f"Loaded {total_cases} test cases.\n")

    resources = _load_resources(config)
    state = RunnerState(
        config=config,
        resources=resources,
        batch=BatchConfig(
            cache_dir=cache_dir,
            resume=resume,
            total_cases=total_cases,
        ),
    )
    stats = RunStats()

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        case = CaseInput(
            index=index,
            hpo_terms=hpo_terms,
            ground_truth=ground_truth,
        )
        _handle_case(case, state, stats)

    print_summary(
        total_cases,
        stats.processed,
        stats.skipped,
        stats.failed,
        stats.total_time,
        cache_dir,
    )
    return cache_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RareSim set-based similarity batch runner",
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
