"""Batch runner for RareSim direct LLM disease retrieval.

Usage:
    python -m scripts.evaluation.run_llm \
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
from raresim.similarity_methods.llm.config import LLM_MODEL_LIST
from raresim.similarity_methods.llm.config import MAX_NEW_TOKENS_RETRIEVAL
from raresim.similarity_methods.llm.methods import (
    build_retrieval_prompt,
    load_hf_pipeline,
    parse_retrieval_output,
    query_hf,
    unload_pipeline,
)
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import preprocess_ancestor_sets
from raresim.utils.io import load_json
from raresim.utils.paths import HPO_LABELS_PATH
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
)


@dataclass
class LlmResources:
    """Shared data needed by the LLM runner."""

    ctx: AppContext
    hpo_labels: Any
    ancestor_sets: dict[str, Any]


@dataclass
class BatchConfig:
    """Static batch-run configuration."""

    cache_dir: Path
    resume: bool
    top_k: int
    total_cases: int


@dataclass
class RunStats:
    """Mutable counters for the batch run."""

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
class CaseExecution:
    """Runtime context for one model/case execution."""

    model_name: str
    pipe: Any
    resources: LlmResources
    batch: BatchConfig
    case: CaseInput


def _load_resources() -> LlmResources:
    """Load shared labels, context, and HPO ancestor sets."""
    hpo_labels = load_json(HPO_LABELS_PATH)
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)

    return LlmResources(
        ctx=ctx,
        hpo_labels=hpo_labels,
        ancestor_sets=ancestor_sets,
    )


def _build_patient(
    case: CaseInput,
    ancestor_sets: dict[str, Any],
) -> PatientProfile:
    """Build a propagated PatientProfile for one evaluation case."""
    return build_patient(
        case.index,
        case.hpo_terms,
        ancestor_sets,
    )


def _serialize_results(results: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert SimilarityResult objects to JSON-serializable dictionaries."""
    serialized: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, dict):
            serialized.append(cast(dict[str, Any], result))
            continue

        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "LLM result must be a SimilarityResult-like object or dict, "
                f"got {type(result).__name__}"
            )

        result_dict = to_dict()
        if not isinstance(result_dict, dict):
            raise TypeError(
                "LLM result.to_dict() must return a dict, "
                f"got {type(result_dict).__name__}"
            )

        serialized.append(cast(dict[str, Any], result_dict))

    return serialized


def _run_single_case(execution: CaseExecution) -> tuple[list[dict[str, Any]], float]:
    """Run one LLM model on one test case."""
    patient = _build_patient(
        execution.case,
        execution.resources.ancestor_sets,
    )
    case_timer = Timer(execution.model_name).start()

    prompt = build_retrieval_prompt(
        patient=patient,
        hpo_labels=execution.resources.hpo_labels,
        top_k=execution.batch.top_k,
    )

    generated_text = query_hf(
        prompt,
        execution.pipe,
        max_tokens=MAX_NEW_TOKENS_RETRIEVAL,
    )

    model_results = parse_retrieval_output(
        generated_text=generated_text,
        patient=patient,
        hpo_labels=execution.resources.hpo_labels,
        disease_profiles=execution.resources.ctx.disease_profiles,
        model_name=execution.model_name,
        top_k=execution.batch.top_k,
        ic_values=execution.resources.ctx.ic_values,
        disease_ancestors=execution.resources.ctx.disease_ancestors,
        disease_metadata_index=execution.resources.ctx.disease_metadata_index,
    )

    elapsed = round(case_timer.stop(), 3)
    return _serialize_results(model_results), elapsed


def _write_error(cache_dir: Path, case_index: int, error: Exception) -> None:
    """Write one case error file."""
    error_path = cache_dir / f"case_{case_index:04d}.error"
    error_path.write_text(
        f"{type(error).__name__}: {error}",
        encoding="utf-8",
    )


def _handle_case(
    execution: CaseExecution,
    stats: RunStats,
) -> None:
    """Run and cache one case, updating batch statistics."""
    cache_file = cache_path_for(execution.batch.cache_dir, execution.case.index)

    if execution.batch.resume and methods_already_cached(
        cache_file,
        [execution.model_name],
    ):
        stats.skipped += 1
        return

    print_case(
        execution.case.index,
        execution.batch.total_cases,
        execution.case.hpo_terms,
        execution.case.ground_truth,
    )

    try:
        serialized_results, elapsed = _run_single_case(execution)
        stats.total_time += elapsed

        save_cache(
            cache_file,
            execution.case.index,
            execution.case.hpo_terms,
            execution.case.ground_truth,
            {execution.model_name: serialized_results},
            {execution.model_name: elapsed},
            elapsed,
        )

        stats.processed += 1
        remaining = execution.batch.total_cases - execution.case.index - 1
        print_case_ok(elapsed, stats.total_time, stats.processed, remaining)

    except Exception as error:
        stats.failed += 1
        print_case_err(error)
        _write_error(
            execution.batch.cache_dir,
            execution.case.index,
            error,
        )


def _run_model_cases(
    model_name: str,
    cases: list[tuple[list[str], list[str]]],
    resources: LlmResources,
    batch: BatchConfig,
    stats: RunStats,
) -> None:
    """Load one model, run it over all cases, then unload it."""
    print(f"\n[llm] Loading model: {model_name}")

    pipe = None
    try:
        pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_RETRIEVAL)

        for index, (hpo_terms, ground_truth) in enumerate(cases):
            case = CaseInput(
                index=index,
                hpo_terms=hpo_terms,
                ground_truth=ground_truth,
            )
            execution = CaseExecution(
                model_name=model_name,
                pipe=pipe,
                resources=resources,
                batch=batch,
                case=case,
            )
            _handle_case(execution, stats)

    finally:
        if pipe is not None:
            unload_pipeline(pipe)


def run(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """Run all configured LLM models on every test case."""
    cache_dir = EVALUATION_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("llm", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    estimated_minutes = total_cases * len(LLM_MODEL_LIST) * 3

    print(f"Loaded {total_cases} test cases.")
    print(f"Models  : {LLM_MODEL_LIST}")
    print(f"Warning : LLM is slow. Estimated total: {estimated_minutes} min\n")

    resources = _load_resources()
    batch = BatchConfig(
        cache_dir=cache_dir,
        resume=resume,
        top_k=top_k,
        total_cases=total_cases,
    )
    stats = RunStats()

    for model_name in LLM_MODEL_LIST:
        _run_model_cases(model_name, cases, resources, batch, stats)

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
        description="RareSim LLM batch runner",
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
