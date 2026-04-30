"""
Result cache for comparing similarity methods across runs.

Saves ranked outputs from all pipelines together in a single file
per run, so results can be compared later without rerunning.

Cache location: outputs/cache/
Cache file naming: {patient_id}_{timestamp}.json

Usage:
    from shared.cache import save_run_cache, load_run_cache, list_cached_runs

    # After running all pipelines — save everything together
    save_run_cache(
        patient_id=patient.patient_id,
        config=config,
        similarity_results=all_results,         # dict[str, list[SimilarityResult]]
        raw_results=all_raw_results,            # dict[str, list[dict]] for transformer/llm
    )

    # Load a previous run for comparison
    cache = load_run_cache(cache_path)

    # List all cached runs for a patient
    runs = list_cached_runs(patient_id="patient_001")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.paths import OUTPUTS_DIR
from shared.result import SimilarityResult

CACHE_DIR = OUTPUTS_DIR / "cache"


# ── Save ──────────────────────────────────────────────────────────────────────


def _serialize_similarity_results(
    results: dict[str, list[SimilarityResult]],
) -> dict[str, list[dict]]:
    """Convert SimilarityResult objects to dicts for JSON serialization."""
    return {
        method: [r.to_dict() for r in rows]
        for method, rows in results.items()
    }


def save_run_cache(
    patient_id: str,
    config,
    similarity_results: dict[str, list[SimilarityResult]] | None = None,
    raw_results: dict[str, list[dict]] | None = None,
    app_metadata: dict | None = None,
) -> Path:
    """
    Save all pipeline results for one run to a single cache file.

    Args:
        patient_id:          Patient identifier.
        config:              PipelineConfig used for this run.
        similarity_results:  Results from semantic/set_based/tfidf pipelines
                             (dict[method_name, list[SimilarityResult]]).
        raw_results:         Results from transformer/llm pipelines
                             (dict[method_name, list[dict]]).
        app_metadata:        AppMetadata dict (n_diseases, n_patient_terms etc.).

    Returns:
        Path to the saved cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{patient_id}_{timestamp}.json"
    cache_path = CACHE_DIR / filename

    # serialize SimilarityResult objects
    serialized_similarity = _serialize_similarity_results(similarity_results or {})

    # merge all results under one dict
    all_results = {**serialized_similarity, **(raw_results or {})}

    cache = {
        "patient_id": patient_id,
        "run_timestamp": timestamp,
        "config": {
            "top_k": config.top_k,
            "use_propagated_terms": config.use_propagated_terms,
            "ic_threshold": config.ic_threshold,
            "use_canonical_profiles": config.use_canonical_profiles,
        },
        "app_metadata": app_metadata or {},
        "methods_run": sorted(all_results.keys()),
        "results": all_results,
    }

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"[cache] Run saved to: {cache_path}")
    return cache_path


# ── Load ──────────────────────────────────────────────────────────────────────


def load_run_cache(cache_path: Path) -> dict:
    """
    Load a cached run from disk.

    Returns the full cache dict with keys:
        patient_id, run_timestamp, config, app_metadata,
        methods_run, results
    """
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── List ──────────────────────────────────────────────────────────────────────


def list_cached_runs(patient_id: Optional[str] = None) -> list[Path]:
    """
    List all cached run files, optionally filtered by patient ID.

    Returns paths sorted by timestamp descending (most recent first).
    """
    if not CACHE_DIR.exists():
        return []

    files = sorted(CACHE_DIR.glob("*.json"), reverse=True)

    if patient_id:
        files = [f for f in files if f.name.startswith(f"{patient_id}_")]

    return files


def print_cached_runs(patient_id: Optional[str] = None) -> None:
    """Print available cached runs in a readable format."""
    runs = list_cached_runs(patient_id)

    if not runs:
        print("[cache] No cached runs found.")
        return

    print(f"\n{'─' * 64}")
    print(f"  Cached runs{f' for {patient_id}' if patient_id else ''}")
    print(f"{'─' * 64}")

    for path in runs:
        try:
            cache = load_run_cache(path)
            methods = ", ".join(cache.get("methods_run", []))
            print(
                f"  {cache['run_timestamp']} | "
                f"{cache['patient_id']} | "
                f"methods: {methods}"
            )
            print(f"    → {path}")
        except Exception:
            print(f"  [unreadable] {path}")
            