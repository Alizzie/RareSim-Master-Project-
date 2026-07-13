"""Batch runner for penalized Jaccard similarity with excluded HPO terms.

The test set must use the negative-aware format:

[
    {
        "hpo_terms": ["HP:..."],
        "excluded_hpo_terms": ["HP:..."],
        "disease_codes": ["ORPHA:...", "OMIM:..."]
    }
]

By default, a test-set filename ending in ``_with_excluded`` writes to the
cache directory of the corresponding positive-only dataset. For example:

    0.1.27_with_excluded.json
        -> outputs/evaluation/0.1.27/cache/

Use ``--cache-name`` to override the target dataset cache explicitly.

Usage:

    python -m scripts.evaluation.run_set_jaccard_penalized \
        --test-set <negative_aware_test_set.json> \
        [--cache-name <existing_cache_dataset_name>] \
        [--no-resume] \
        [--limit <max_cases>] \
        [--top-k <top_k_results>]
"""

# pylint: disable=broad-exception-caught,too-few-public-methods

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raresim.core import AppContext, PipelineConfig
from raresim.similarity_methods.set_based import (
    METHOD_NAMES,
    run as run_set_based,
)
from raresim.types import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    add_common_args,
    build_patient,
    cache_path_for,
    methods_already_cached,
    print_case,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
    serialize_results,
)


PENALIZED_METHOD = "set_jaccard_penalized"
REQUIRED_METHODS = [PENALIZED_METHOD]
EXCLUDED_SUFFIX = "_with_excluded"


@dataclass(frozen=True)
class SharedResources:
    """Shared RareSim context and preprocessed ontology data."""

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
    """One negative-aware evaluation case."""

    index: int
    hpo_terms: list[str]
    excluded_hpo_terms: list[str]
    ground_truth: list[str]


@dataclass(frozen=True)
class RunnerState:
    """State required to run one penalized Jaccard evaluation case."""

    config: PipelineConfig
    resources: SharedResources
    batch: BatchConfig


def _validate_method_registration() -> None:
    """Ensure the penalized method is registered in the set-based pipeline."""
    if PENALIZED_METHOD not in METHOD_NAMES:
        raise ValueError(
            f"{PENALIZED_METHOD!r} is not registered in METHOD_NAMES. "
            "Check raresim.similarity_methods.set_based.config."
        )


def _normalize_string_list(
    value: Any,
    field_name: str,
    case_index: int,
    *,
    allow_empty: bool,
) -> list[str]:
    """Validate and normalize one list-valued test-case field."""
    if not isinstance(value, list):
        raise ValueError(
            f"Case {case_index}: {field_name!r} must be a list, "
            f"got {type(value).__name__}."
        )

    if any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"Case {case_index}: {field_name!r} must contain only strings."
        )

    normalized = sorted({item.strip() for item in value if item.strip()})

    if not allow_empty and not normalized:
        raise ValueError(
            f"Case {case_index}: {field_name!r} must not be empty."
        )

    return normalized


def load_negative_aware_test_cases(test_set_path: Path) -> list[CaseInput]:
    """Load dictionary-based cases containing positive and excluded HPO terms."""
    data = json.loads(test_set_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {test_set_path}, "
            f"got {type(data).__name__}."
        )

    cases: list[CaseInput] = []

    for index, raw_case in enumerate(data):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Case {index}: expected a JSON object, "
                f"got {type(raw_case).__name__}."
            )

        hpo_terms = _normalize_string_list(
            raw_case.get("hpo_terms"),
            "hpo_terms",
            index,
            allow_empty=False,
        )
        excluded_hpo_terms = _normalize_string_list(
            raw_case.get("excluded_hpo_terms", []),
            "excluded_hpo_terms",
            index,
            allow_empty=True,
        )
        ground_truth = _normalize_string_list(
            raw_case.get("disease_codes"),
            "disease_codes",
            index,
            allow_empty=False,
        )

        cases.append(
            CaseInput(
                index=index,
                hpo_terms=hpo_terms,
                excluded_hpo_terms=excluded_hpo_terms,
                ground_truth=ground_truth,
            )
        )

    return cases


def _resolve_cache_name(test_set_path: Path, cache_name: str | None) -> str:
    """Resolve the dataset cache name used for incremental result merging."""
    if cache_name:
        return cache_name

    stem = test_set_path.stem
    if stem.endswith(EXCLUDED_SUFFIX):
        return stem[: -len(EXCLUDED_SUFFIX)]

    return stem


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


def _build_negative_aware_patient(
    case: CaseInput,
    ancestor_sets: dict[str, Any],
) -> PatientProfile:
    """Build a patient while preserving the shared positive-term logic."""
    base_patient = build_patient(
        case.index,
        case.hpo_terms,
        ancestor_sets,
    )

    return PatientProfile(
        patient_id=base_patient.patient_id,
        raw_text=base_patient.raw_text,
        hpo_terms=set(base_patient.hpo_terms),
        propagated_hpo_terms=set(base_patient.propagated_hpo_terms),
        excluded_hpo_terms=set(case.excluded_hpo_terms),
    )


def _write_error(cache_dir: Path, case_index: int, error: Exception) -> None:
    """Write one case error file."""
    error_path = cache_dir / f"case_{case_index:04d}.error"
    error_path.write_text(
        f"{type(error).__name__}: {error}",
        encoding="utf-8",
    )


def _validate_existing_cache_alignment(
    cache_file: Path,
    case: CaseInput,
) -> None:
    """Prevent results from being merged into a mismatched case cache."""
    if not cache_file.exists():
        return

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(cached, dict):
        raise ValueError(f"Existing cache is not a JSON object: {cache_file}")

    cached_index = cached.get("case_index")
    if cached_index is not None and cached_index != case.index:
        raise ValueError(
            f"Cache alignment error for {cache_file}: "
            f"case_index is {cached_index}, expected {case.index}."
        )

    cached_hpo_terms = cached.get("hpo_terms")
    if cached_hpo_terms is not None:
        if not isinstance(cached_hpo_terms, list):
            raise ValueError(
                f"Cache alignment error for {cache_file}: "
                "hpo_terms is not a list."
            )

        if set(cached_hpo_terms) != set(case.hpo_terms):
            raise ValueError(
                f"Cache alignment error for {cache_file}: "
                "positive HPO terms do not match the negative-aware test set."
            )

    cached_ground_truth = cached.get("ground_truth")
    if cached_ground_truth is not None:
        if not isinstance(cached_ground_truth, list):
            raise ValueError(
                f"Cache alignment error for {cache_file}: "
                "ground_truth is not a list."
            )

        if set(cached_ground_truth) != set(case.ground_truth):
            raise ValueError(
                f"Cache alignment error for {cache_file}: "
                "ground-truth disease codes do not match."
            )


def _save_case_cache(
    cache_file: Path,
    case: CaseInput,
    serialized_results: dict[str, list[dict[str, Any]]],
    method_elapsed: dict[str, float],
    elapsed: float,
) -> None:
    """Merge the method result and preserve excluded terms in the cache."""
    save_cache(
        cache_file,
        case.index,
        case.hpo_terms,
        case.ground_truth,
        serialized_results,
        method_elapsed,
        elapsed,
    )

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    if not isinstance(cached, dict):
        raise ValueError(f"Saved cache is not a JSON object: {cache_file}")

    cached["excluded_hpo_terms"] = case.excluded_hpo_terms
    cache_file.write_text(
        json.dumps(cached, indent=2),
        encoding="utf-8",
    )


def _run_case(
    case: CaseInput,
    state: RunnerState,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], float]:
    """Run only penalized Jaccard for one evaluation case."""
    patient = _build_negative_aware_patient(
        case,
        state.resources.ancestor_sets,
    )

    case_timer = Timer(PENALIZED_METHOD).start()
    method_timer = Timer(PENALIZED_METHOD).start()

    method_results = run_set_based(
        patient,
        REQUIRED_METHODS,
        state.config,
        state.resources.ctx,
    )

    method_elapsed = {
        PENALIZED_METHOD: round(method_timer.stop(), 3),
    }
    elapsed = round(case_timer.stop(), 3)

    return serialize_results(method_results), method_elapsed, elapsed


def _handle_case(
    case: CaseInput,
    state: RunnerState,
    stats: RunStats,
) -> None:
    """Run, cache, and log one evaluation case."""
    cache_file = cache_path_for(state.batch.cache_dir, case.index)

    _validate_existing_cache_alignment(cache_file, case)

    if state.batch.resume and methods_already_cached(
        cache_file,
        REQUIRED_METHODS,
    ):
        stats.skipped += 1
        return

    print_case(
        case.index,
        state.batch.total_cases,
        case.hpo_terms,
        case.ground_truth,
    )
    print(f"  Excluded HPO terms: {len(case.excluded_hpo_terms)}")

    try:
        serialized_results, method_elapsed, elapsed = _run_case(case, state)
        stats.total_time += elapsed

        _save_case_cache(
            cache_file,
            case,
            serialized_results,
            method_elapsed,
            elapsed,
        )

        stats.processed += 1
        remaining = state.batch.total_cases - case.index - 1
        print_case_ok(
            elapsed,
            stats.total_time,
            stats.processed,
            remaining,
        )

    except Exception as error:
        stats.failed += 1
        print_case_err(error)
        _write_error(state.batch.cache_dir, case.index, error)


def run(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
    cache_name: str | None = None,
) -> Path:
    """Run penalized Jaccard on every negative-aware test case."""
    _validate_method_registration()

    if config is None:
        config = PipelineConfig(
            use_propagated_terms=True,
            use_canonical_profiles=True,
        )

    resolved_cache_name = _resolve_cache_name(test_set_path, cache_name)
    cache_dir = EVALUATION_DIR / resolved_cache_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(
        PENALIZED_METHOD,
        test_set_path,
        cache_dir,
        resume,
        limit,
    )

    cases = load_negative_aware_test_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    print(f"Loaded {total_cases} negative-aware test cases.\n")

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

    for case in cases:
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
        description=(
            "Run RareSim penalized Jaccard with positive and excluded "
            "patient HPO terms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--cache-name",
        default=None,
        help=(
            "Dataset cache directory name. By default, a trailing "
            "'_with_excluded' is removed from the test-set filename."
        ),
    )
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
        cache_name=args.cache_name,
    )


if __name__ == "__main__":
    main()
