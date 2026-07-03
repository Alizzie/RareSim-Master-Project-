"""
raresim.types — shared data contracts used across the whole codebase.

These are the stable structures that pipelines, explanation builders,
caching, and analysis all depend on. Re-exported here so callers can write
    from raresim.types import PatientProfile, SimilarityResult
instead of reaching into individual submodules.

Two groups:
    schemas.py — input/domain entities (PatientProfile, DiseaseProfile)
    result.py  — pipeline output structures (SimilarityResult, MethodResults, ...)
"""

# ── Domain / input entities ───────────────────────────────────────────────────
from raresim.types.schemas import (
    PatientProfile,
    DiseaseProfile,
)

# ── Pipeline result structures ────────────────────────────────────────────────
from raresim.types.result import (
    SimilarityResult,
    MethodResults,
    RunStats,
    AppMetadata,
    SCHEMA_VERSION,
    PipelineConfig,
)

__all__ = [
    # schemas
    "PatientProfile",
    "DiseaseProfile",
    # results
    "SimilarityResult",
    "MethodResults",
    "RunStats",
    "AppMetadata",
    "SCHEMA_VERSION",
    "PipelineConfig",
]
