"""Batch runner for RareSim TF-IDF text similarity.

The runner uses raw clinical text as the patient query and runs only the
TF-IDF method whose registered name represents text mode.

Supported test-set formats
--------------------------

1. List of objects:

[
    {
        "id": "case_0000",
        "raw_text": "Clinical description...",
        "disease_codes": ["ORPHA:123"]
    }
]

The following aliases are also accepted:

- Text: ``raw_text``, ``text``, ``clinical_text``, or ``description``
- Ground truth: ``disease_codes``, ``ground_truth``, ``disease_id``,
  ``disease_code``, or ``orpha_code``

2. Disease-to-text mapping:

{
    "ORPHA:123": "Clinical description...",
    "ORPHA:456": "Another clinical description..."
}

Usage:

    python -m scripts.evaluation.run_tfidf_text \
        --test-set data/datasets/free_text/medicalCases_200.json \
        [--cache-name medical_cases_raw] \
        [--method tfidf_text] \
        [--no-resume] \
        [--limit 10] \
        [--top-k 10]
"""

# pylint: disable=broad-exception-caught,too-few-public-methods

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raresim.core import AppContext, PipelineConfig
from raresim.similarity_methods.tfidf import (
    METHOD_NAMES,
    run as run_tfidf,
)
from raresim.types import PatientProfile
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
    serialize_results,
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


@dataclass(frozen=True)
class SharedResources:
    """Shared RareSim application context."""

    ctx: AppContext


@dataclass(frozen=True)
class BatchConfig:
    """Static batch-run configuration."""

    cache_dir: Path
    resume: bool
    total_cases: int
    method_name: str


@dataclass
class RunStats:
    """Mutable batch-run counters."""

    skipped: int = 0
    processed: int = 0
    failed: int = 0
    total_time: float = 0.0


@dataclass(frozen=True)
class RunnerState:
    """State required to run one raw-text TF-IDF case."""

    config: PipelineConfig
    resources: SharedResources
    batch: BatchConfig


def _first_present(
    entry: dict[str, Any],
    field_names: tuple[str, ...],
) -> Any:
    """Return the first present field value from a sequence of aliases."""
    for field_name in field_names:
        if field_name in entry:
            return entry[field_name]
    return None


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

    normalized = sorted({item.strip() for item in values if item.strip()})

    if not normalized:
        raise ValueError(f"Case {case_index}: ground truth is empty.")

    return normalized


def _load_mapping_cases(data: dict[str, Any]) -> list[RawTextCase]:
    """Load ``disease_id -> clinical text`` mapping cases."""
    cases: list[RawTextCase] = []

    for index, (disease_id, raw_text) in enumerate(data.items()):
        if not isinstance(disease_id, str) or not disease_id.strip():
            raise ValueError(
                f"Case {index}: disease mapping key must be a non-empty string."
            )

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                f"Case {index}: mapped clinical text must be non-empty."
            )

        normalized_disease_id = disease_id.strip()
        if normalized_disease_id.isdigit():
            normalized_disease_id = f"ORPHA:{normalized_disease_id}"

        cases.append(
            RawTextCase(
                index=index,
                case_id=f"case_{index:04d}",
                raw_text=raw_text.strip(),
                ground_truth=[normalized_disease_id],
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
                f"Case {index}: no non-empty raw-text field was found. "
                f"Accepted fields: {TEXT_FIELD_NAMES}."
            )

        ground_truth_value = _first_present(
            entry,
            GROUND_TRUTH_FIELD_NAMES,
        )
        ground_truth = _normalize_ground_truth(
            ground_truth_value,
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
    """Load a supported raw-text test-set format."""
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


def _normalized_method_name(method_name: str) -> str:
    """Normalize a method name for robust text-mode detection."""
    return "".join(
        character
        for character in method_name.lower()
        if character.isalnum()
    )


def resolve_text_method(requested_method: str | None) -> str:
    """Resolve the registered TF-IDF text-mode method name."""
    registered = list(METHOD_NAMES)

    if requested_method is not None:
        if requested_method not in registered:
            raise ValueError(
                f"Unknown TF-IDF method {requested_method!r}. "
                f"Registered methods: {registered}"
            )
        return requested_method

    if "tfidf_text" in registered:
        return "tfidf_text"

    normalized_matches = [
        method
        for method in registered
        if _normalized_method_name(method) == "tfidftext"
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    text_matches = [
        method
        for method in registered
        if "tfidf" in _normalized_method_name(method)
        and "text" in _normalized_method_name(method)
    ]
    if len(text_matches) == 1:
        return text_matches[0]

    raise ValueError(
        "Could not identify one TF-IDF text method automatically. "
        f"Registered methods: {registered}. "
        "Pass the exact name with --method."
    )


def _load_resources(config: PipelineConfig) -> SharedResources:
    """Load the shared RareSim context."""
    print("Loading shared context...")

    dummy = PatientProfile(
        patient_id="batch_init",
        raw_text="",
        hpo_terms=set(),
        propagated_hpo_terms=set(),
    )
    ctx = AppContext.load(
        dummy,
        use_canonical_profiles=config.use_canonical_profiles,
    )

    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print(f"  HPO labels       : {ctx.app_metadata.n_hpo_labels}")
    print("  Ready.\n")

    return SharedResources(ctx=ctx)


def _build_patient(case: RawTextCase) -> PatientProfile:
    """Build a raw-text-only patient profile."""
    return PatientProfile(
        patient_id=case.case_id,
        raw_text=case.raw_text,
        hpo_terms=set(),
        propagated_hpo_terms=set(),
    )


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


def _save_raw_text_cache(
    cache_file: Path,
    case: RawTextCase,
    serialized_results: dict[str, list[dict[str, Any]]],
    method_elapsed: dict[str, float],
    elapsed: float,
) -> None:
    """Merge results and add raw-text metadata to the case cache."""
    save_cache(
        cache_file,
        case.index,
        [],
        case.ground_truth,
        serialized_results,
        method_elapsed,
        elapsed,
    )

    cached = load_cache(cache_file)
    cached["case_id"] = case.case_id
    cached["raw_text"] = case.raw_text

    cache_file.write_text(
        json.dumps(cached, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_case(
    case: RawTextCase,
    state: RunnerState,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], float]:
    """Run TF-IDF text similarity for one raw-text case."""
    patient = _build_patient(case)
    method_name = state.batch.method_name

    case_timer = Timer(method_name).start()
    method_timer = Timer(method_name).start()

    method_results = run_tfidf(
        patient,
        [method_name],
        state.config,
        state.resources.ctx,
    )

    method_elapsed = {
        method_name: round(method_timer.stop(), 3),
    }
    elapsed = round(case_timer.stop(), 3)

    return serialize_results(method_results), method_elapsed, elapsed


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


def _handle_case(
    case: RawTextCase,
    state: RunnerState,
    stats: RunStats,
) -> None:
    """Run, cache, and log one raw-text case."""
    cache_file = cache_path_for(state.batch.cache_dir, case.index)
    required_methods = [state.batch.method_name]

    if state.batch.resume and methods_already_cached(
        cache_file,
        required_methods,
    ):
        stats.skipped += 1
        return

    _print_raw_text_case(case, state.batch.total_cases)

    try:
        serialized_results, method_elapsed, elapsed = _run_case(
            case,
            state,
        )
        stats.total_time += elapsed

        _save_raw_text_cache(
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


def run(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
    cache_name: str | None = None,
    method: str | None = None,
) -> Path:
    """Run TF-IDF text similarity on every raw-text test case."""
    method_name = resolve_text_method(method)

    if config is None:
        config = PipelineConfig(
            use_propagated_terms=False,
            use_canonical_profiles=True,
        )

    resolved_cache_name = cache_name or test_set_path.stem
    cache_dir = EVALUATION_DIR / resolved_cache_name / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(
        method_name,
        test_set_path,
        cache_dir,
        resume,
        limit,
    )

    cases = load_raw_text_cases(test_set_path)
    if limit is not None:
        cases = cases[:limit]

    total_cases = len(cases)
    print(f"Loaded {total_cases} raw-text test cases.\n")
    print(f"TF-IDF method: {method_name}\n")

    resources = _load_resources(config)
    state = RunnerState(
        config=config,
        resources=resources,
        batch=BatchConfig(
            cache_dir=cache_dir,
            resume=resume,
            total_cases=total_cases,
            method_name=method_name,
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
        description="RareSim raw-text TF-IDF batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument(
        "--cache-name",
        default=None,
        help=(
            "Optional evaluation cache directory name. "
            "Defaults to the test-set filename stem."
        ),
    )
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Exact registered TF-IDF text method name. "
            "Normally detected automatically."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    config = PipelineConfig(
        top_k=args.top_k,
        use_propagated_terms=False,
        use_canonical_profiles=True,
    )
    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        config=config,
        limit=args.limit,
        cache_name=args.cache_name,
        method=args.method,
    )


if __name__ == "__main__":
    main()
