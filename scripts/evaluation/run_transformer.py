"""Batch runner for RareSim transformer retrieval methods.
Usage:
    python -m scripts.evaluation.run_transformer \
        --test-set <path_to_test_set.json> \
        [--no-resume] \
        [--limit <max_cases>] \
        [--top-k <top_k_results>]
"""

# pylint: disable=broad-exception-caught,too-few-public-methods

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from raresim.core.context import AppContext
from raresim.similarity_methods.transformer.config import CANDIDATE_POOL_SIZE
from raresim.similarity_methods.transformer.config import MODEL_LIST
from raresim.similarity_methods.transformer.retriever import DiseaseRetriever
from raresim.types.schemas import PatientProfile
from raresim.utils.io import load_json
from raresim.utils.paths import ALIAS_TO_CANONICAL_PATH
from raresim.utils.paths import HPO_LABELS_PATH
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


@dataclass(frozen=True)
class TransformerResources:
    """Shared transformer retriever resources."""

    retriever: DiseaseRetriever


@dataclass(frozen=True)
class BatchConfig:
    """Static batch-run configuration."""

    cache_dir: Path
    resume: bool
    top_k: int
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
    """State needed to run one transformer evaluation case."""

    resources: TransformerResources
    batch: BatchConfig


def _build_patient(case: CaseInput) -> PatientProfile:
    """Build the PatientProfile expected by the transformer retriever."""
    return PatientProfile(
        patient_id=f"eval_case_{case.index:04d}",
        raw_text="",
        hpo_terms=set(case.hpo_terms),
        propagated_hpo_terms=set(case.hpo_terms),
    )


def _serialize_results(results: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert result objects to JSON-serializable dictionaries."""
    serialized: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, dict):
            serialized.append(cast(dict[str, Any], result))
            continue

        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "Transformer result must be a dict or SimilarityResult-like object, "
                f"got {type(result).__name__}"
            )

        result_dict = to_dict()
        if not isinstance(result_dict, dict):
            raise TypeError(
                "Transformer result.to_dict() must return a dict, "
                f"got {type(result_dict).__name__}"
            )

        serialized.append(cast(dict[str, Any], result_dict))

    return serialized


def _load_resources() -> TransformerResources:
    """Load shared context and prepare the transformer retriever."""
    hpo_labels = load_json(HPO_LABELS_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)

    dummy_patient = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy_patient, use_canonical_profiles=True)

    print(f"Models: {MODEL_LIST}")
    print("Preparing transformer embedding cache...")

    retriever = DiseaseRetriever(
        patient=dummy_patient,
        disease_profiles=ctx.disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=MODEL_LIST,
    )
    retriever.warmup(preload_models=False)

    print("  Ready.\n")
    return TransformerResources(retriever=retriever)


def _run_case(
    case: CaseInput,
    state: RunnerState,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], float]:
    """Run all transformer models on one case."""
    all_results: dict[str, list[dict[str, Any]]] = {}
    method_elapsed: dict[str, float] = {}
    patient = _build_patient(case)

    case_timer = Timer("transformer").start()

    for model_name in MODEL_LIST:
        model_timer = Timer(model_name).start()

        model_results = state.resources.retriever.rank(
            model_name=model_name,
            patient=patient,
            top_k=state.batch.top_k,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
        )

        method_elapsed[model_name] = round(model_timer.stop(), 3)
        all_results[model_name] = _serialize_results(model_results)

    elapsed = round(case_timer.stop(), 3)
    return all_results, method_elapsed, elapsed


def _write_error(cache_dir: Path, case_index: int, error: Exception) -> None:
    """Write one case error file."""
    error_path = cache_dir / f"case_{case_index:04d}.error"
    error_path.write_text(
        f"{type(error).__name__}: {error}",
        encoding="utf-8",
    )


def _handle_case(
    case: CaseInput,
    state: RunnerState,
    stats: RunStats,
) -> None:
    """Run, cache, and log one evaluation case."""
    cache_file = cache_path_for(state.batch.cache_dir, case.index)

    if state.batch.resume and methods_already_cached(cache_file, MODEL_LIST):
        stats.skipped += 1
        return

    print_case(
        case.index,
        state.batch.total_cases,
        case.hpo_terms,
        case.ground_truth,
    )

    try:
        all_results, method_elapsed, elapsed = _run_case(case, state)
        stats.total_time += elapsed

        save_cache(
            cache_file,
            case.index,
            case.hpo_terms,
            case.ground_truth,
            all_results,
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
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """Run all configured transformer models on every test case."""
    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("transformer", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    print(f"Loaded {total_cases} test cases.\n")

    state = RunnerState(
        resources=_load_resources(),
        batch=BatchConfig(
            cache_dir=cache_dir,
            resume=resume,
            top_k=top_k,
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
        description="RareSim transformer batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        limit=args.limit,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
