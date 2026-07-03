"""
raresim.core — foundational infrastructure for similarity pipelines.

This package holds what every method needs to run: pipeline configuration,
the app context, result types, caching, and the explanation schema.

Re-exports below are the stable, frequently-imported entry points.
Anything not listed here should be imported from its submodule directly
(e.g. explanation builders from raresim.core.explanation).
"""

# ── Configuration and orchestration ──────────────────────────────────────────
from raresim.core.pipeline import (
    PipelineConfig,
    build_run_stats,
    sort_and_rank,
)

# ── Application context ───────────────────────────────────────────────────────
from raresim.core.context import AppContext

# ── Result types (also re-exported from types.result) ────────────────────────
from raresim.types.result import (
    AppMetadata,
    MethodResults,
    RunStats,
    SimilarityResult,
)

# ── Caching ───────────────────────────────────────────────────────────────────
from raresim.core.cache import (
    save_run_cache,
    load_run_cache,
    list_cached_runs,
)

# ── Runner ───────────────────────────────────────────────────────────────────
from raresim.core.method_runner import run_similarity_method

__all__ = [
    # configuration / orchestration
    "PipelineConfig",
    "build_run_stats",
    "sort_and_rank",
    # context
    "AppContext",
    # result types
    "AppMetadata",
    "MethodResults",
    "RunStats",
    "SimilarityResult",
    # caching
    "save_run_cache",
    "load_run_cache",
    "list_cached_runs",
    # runner
    "run_similarity_method",
]
