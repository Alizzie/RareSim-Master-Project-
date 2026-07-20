"""Batch runner for RareSim direct LLM retrieval from raw patient text.

Supported input formats:

{
    "58": "Patient clinical text...",
    "61": "Another patient clinical text..."
}

or:

[
    {
        "id": "case_0000",
        "raw_text": "Patient clinical text...",
        "disease_codes": ["ORPHA:58"]
    }
]

Usage:

    python -m scripts.evaluation.run_llm_text \
        --test-set data/datasets/free_text/medicalCases_200.json \
        [--cache-name medical_cases_raw] \
        [--no-resume] \
        [--limit 10] \
        [--top-k 10]
"""

# pylint: disable=broad-exception-caught,too-few-public-methods

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from raresim.core.context import AppContext
from raresim.similarity_methods.llm.config import (
    LLM_MODEL_LIST,
    MAX_NEW_TOKENS_RETRIEVAL,
)
from raresim.similarity_methods.llm.methods import (
    build_retrieval_prompt,
    load_hf_pipeline,
    parse_retrieval_output,
    query_hf,
    unload_pipeline,
)
from raresim.types.schemas import PatientProfile
from raresim.utils.io import load_json
from raresim.utils.paths import HPO_LABELS_PATH
from raresim.utils.timer import Timer

from scripts.evaluation._batch_utils import (
    EVALUATION_DIR,
    add_common_args,
    cache_path_for,
    load_cache,
    methods_already_cached,
    print_case_err,
    print_case_ok,
    print_header,
    print_summary,
    save_cache,
)

TEXT_FIELD_NAMES = (
    "raw_text",
    "text",
    "clinical_text",
    "description",
)

GROUND_TRUTH_FIELD_NAMES = (
    "disease_codes",
    "ground_truth",
    "disease_id",
    "disease_code",
    "orpha_code",
)


@dataclass(frozen=True)
class RawTextCase:
    """One raw-text evaluation case."""

    index: int
    case_id: str
    raw_text: str
    ground_truth: list[str]


def _first_present(
    entry: dict[str, Any],
    field_names: tuple[str, ...],
) -> Any:
    """Return the first present value among supported aliases."""
    for field_name in field_names:
        if field_name in entry:
            return entry[field_name]
    return None


def _normalize_disease_id(disease_id: str) -> str:
    """Normalize bare numeric ORPHA identifiers."""
    normalized = disease_id.strip()
    if normalized.isdigit():
        return f"ORPHA:{normalized}"
    return normalized


def _normalize_ground_truth(value: Any, case_index: int) -> list[str]:
    """Normalize one disease identifier or a list of identifiers."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"Case {case_index}: ground truth must be a string or list."
        )

    if any(not isinstance(item, str) for item in values):
        raise ValueError(
            f"Case {case_index}: ground truth must contain only strings."
        )

    normalized = sorted(
        {
            _normalize_disease_id(item)
            for item in values
            if item.strip()
        }
    )
    if not normalized:
        raise ValueError(f"Case {case_index}: ground truth is empty.")

    return normalized


def _load_mapping_cases(data: dict[str, Any]) -> list[RawTextCase]:
    """Load ``ORPHA number -> clinical text`` mapping cases."""
    cases: list[RawTextCase] = []

    for index, (disease_id, raw_text) in enumerate(data.items()):
        if not isinstance(disease_id, str) or not disease_id.strip():
            raise ValueError(
                f"Case {index}: disease mapping key must be non-empty."
            )

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"Case {index}: mapped clinical text must be non-empty."
            )

        cases.append(
            RawTextCase(
                index=index,
                case_id=f"case_{index:04d}",
                raw_text=raw_text.strip(),
                ground_truth=[_normalize_disease_id(disease_id)],
            )
        )

    return cases


def _load_object_cases(data: list[Any]) -> list[RawTextCase]:
    """Load list-based raw-text cases."""
    cases: list[RawTextCase] = []

    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Case {index}: expected a JSON object, "
                f"got {type(entry).__name__}."
            )

        raw_text = _first_present(entry, TEXT_FIELD_NAMES)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"Case {index}: no valid raw-text field was found."
            )

        ground_truth = _normalize_ground_truth(
            _first_present(entry, GROUND_TRUTH_FIELD_NAMES),
            index,
        )

        case_id_value = entry.get("id", f"case_{index:04d}")
        case_id = (
            case_id_value.strip()
            if isinstance(case_id_value, str) and case_id_value.strip()
            else f"case_{index:04d}"
        )

        cases.append(
            RawTextCase(
                index=index,
                case_id=case_id,
                raw_text=raw_text.strip(),
                ground_truth=ground_truth,
            )
        )

    return cases


def load_raw_text_cases(test_set_path: Path) -> list[RawTextCase]:
    """Load a raw-text test set from a supported JSON structure."""
    if not test_set_path.exists():
        raise FileNotFoundError(
            f"Test set does not exist: {test_set_path}"
        )

    data = json.loads(test_set_path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        cases = _load_mapping_cases(data)
    elif isinstance(data, list):
        cases = _load_object_cases(data)
    else:
        raise ValueError(
            "The test-set root must be a JSON object or JSON list."
        )

    if not cases:
        raise ValueError(f"No cases found in {test_set_path}.")

    return cases


def _build_raw_text_patient(case: RawTextCase) -> PatientProfile:
    """Build a patient profile containing raw text and no HPO terms."""
    return PatientProfile(
        patient_id=case.case_id,
        raw_text=case.raw_text,
        hpo_terms=set(),
        propagated_hpo_terms=set(),
    )


def _validate_existing_cache_alignment(
    cache_file: Path,
    case: RawTextCase,
) -> None:
    """Prevent results from being merged into a different case."""
    if not cache_file.exists():
        return

    cached = load_cache(cache_file)

    cached_ground_truth = cached.get("ground_truth")
    if (
        isinstance(cached_ground_truth, list)
        and set(cached_ground_truth) != set(case.ground_truth)
    ):
        raise ValueError(
            f"Cache alignment error for {cache_file}: "
            "ground truth does not match."
        )

    cached_raw_text = cached.get("raw_text")
    if (
        isinstance(cached_raw_text, str)
        and cached_raw_text.strip() != case.raw_text
    ):
        raise ValueError(
            f"Cache alignment error for {cache_file}: "
            "raw text does not match."
        )


def _save_raw_text_cache(
    cache_file: Path,
    case: RawTextCase,
    results: dict[str, list[dict[str, Any]]],
    method_elapsed: dict[str, float],
    total_elapsed: float,
) -> None:
    """Merge method results and preserve raw-text metadata."""
    save_cache(
        cache_file,
        case.index,
        [],
        case.ground_truth,
        results,
        method_elapsed,
        total_elapsed,
    )

    cached = load_cache(cache_file)
    cached["case_id"] = case.case_id
    cached["raw_text"] = case.raw_text

    cache_file.write_text(
        json.dumps(cached, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _print_raw_text_case(
    case: RawTextCase,
    total_cases: int,
) -> None:
    """Print one raw-text case progress line."""
    preview = " ".join(case.raw_text.split())
    if len(preview) > 80:
        preview = f"{preview[:77]}..."

    print(
        f"[{case.index + 1:>4}/{total_cases}] "
        f"case_{case.index:04d} | "
        f"{len(case.raw_text)} chars | "
        f"gt={case.ground_truth}"
    )
    print(f"           Text: {preview}")



@dataclass(frozen=True)
class LlmResources:
    """Shared data needed by the raw-text LLM runner."""

    ctx: AppContext
    hpo_labels: Any


@dataclass(frozen=True)
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
class CaseExecution:
    """Runtime context for one model/case execution."""

    model_name: str
    pipe: Any
    resources: LlmResources
    batch: BatchConfig
    case: RawTextCase


def _load_resources() -> LlmResources:
    """Load shared HPO labels and application context."""
    hpo_labels = load_json(HPO_LABELS_PATH)
    dummy = PatientProfile(
        patient_id="batch_init",
        raw_text="",
        hpo_terms=set(),
        propagated_hpo_terms=set(),
    )
    ctx = AppContext.load(
        dummy,
        use_canonical_profiles=True,
    )

    return LlmResources(
        ctx=ctx,
        hpo_labels=hpo_labels,
    )


def _serialize_results(results: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert LLM results to JSON-serializable dictionaries."""
    serialized: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, dict):
            serialized.append(cast(dict[str, Any], result))
            continue

        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(
                "LLM result must be a dict or "
                "SimilarityResult-like object, "
                f"got {type(result).__name__}."
            )

        result_dict = to_dict()
        if not isinstance(result_dict, dict):
            raise TypeError(
                "LLM result.to_dict() must return a dict, "
                f"got {type(result_dict).__name__}."
            )

        serialized.append(cast(dict[str, Any], result_dict))

    return serialized


def _run_single_case(
    execution: CaseExecution,
) -> tuple[list[dict[str, Any]], float]:
    """Run one LLM model on one raw-text case."""
    patient = _build_raw_text_patient(execution.case)
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
        disease_metadata_index=(
            execution.resources.ctx.disease_metadata_index
        ),
    )

    elapsed = round(case_timer.stop(), 3)
    return _serialize_results(model_results), elapsed


def _write_error(
    cache_dir: Path,
    case_index: int,
    error: Exception,
) -> None:
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
    """Run and cache one raw-text LLM case."""
    cache_file = cache_path_for(
        execution.batch.cache_dir,
        execution.case.index,
    )
    _validate_existing_cache_alignment(cache_file, execution.case)

    if execution.batch.resume and methods_already_cached(
        cache_file,
        [execution.model_name],
    ):
        stats.skipped += 1
        return

    _print_raw_text_case(
        execution.case,
        execution.batch.total_cases,
    )

    try:
        serialized_results, elapsed = _run_single_case(execution)
        stats.total_time += elapsed

        _save_raw_text_cache(
            cache_file,
            execution.case,
            {execution.model_name: serialized_results},
            {execution.model_name: elapsed},
            elapsed,
        )

        stats.processed += 1
        remaining = (
            execution.batch.total_cases
            - execution.case.index
            - 1
        )
        print_case_ok(
            elapsed,
            stats.total_time,
            stats.processed,
            remaining,
        )

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
    cases: list[RawTextCase],
    resources: LlmResources,
    batch: BatchConfig,
    stats: RunStats,
) -> None:
    """Load one model, run all cases, then unload it."""
    print(f"\\n[llm-text] Loading model: {model_name}")

    pipe = None
    try:
        pipe = load_hf_pipeline(
            model_name,
            MAX_NEW_TOKENS_RETRIEVAL,
        )

        for case in cases:
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
    cache_name: str | None = None,
) -> Path:
    """Run all configured LLM models on raw-text cases."""
    resolved_cache_name = cache_name or test_set_path.stem
    cache_dir = EVALUATION_DIR / resolved_cache_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(
        "llm-text",
        test_set_path,
        cache_dir,
        resume,
        limit,
    )

    cases = load_raw_text_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    print(f"Loaded {total_cases} raw-text test cases.")
    print(f"Models: {LLM_MODEL_LIST}\\n")

    resources = _load_resources()
    batch = BatchConfig(
        cache_dir=cache_dir,
        resume=resume,
        top_k=top_k,
        total_cases=total_cases,
    )
    stats = RunStats()

    for model_name in LLM_MODEL_LIST:
        _run_model_cases(
            model_name,
            cases,
            resources,
            batch,
            stats,
        )

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
        description="RareSim raw-text LLM batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--cache-name",
        default=None,
        help=(
            "Evaluation cache directory name. "
            "Defaults to the test-set filename stem."
        ),
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
        cache_name=args.cache_name,
    )


if __name__ == "__main__":
    main()
