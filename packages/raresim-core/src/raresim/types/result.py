"""
Unified result schema for all similarity pipelines.

Every method returns a MethodResults object containing:
    - RunConfig: static settings used before the run
    - RunStats: observed values computed after the run
    - SimilarityResult list: one entry per ranked disease

AppMetadata is separate because it describes the loaded data, not a pipeline run.
"""

from dataclasses import dataclass, field

SCHEMA_VERSION = "1.1"


@dataclass
class AppMetadata:
    """Metadata describing the loaded application data."""

    n_hpo_labels: int
    n_disease_profiles: int
    unfound_patient_terms: list[str]

    def to_dict(self) -> dict:
        """Return the metadata as a JSON-serializable dictionary."""
        return {
            "n_hpo_labels": self.n_hpo_labels,
            "n_disease_profiles": self.n_disease_profiles,
            "unfound_patient_terms": self.unfound_patient_terms,
        }


@dataclass
class RunConfig:
    """Static configuration used by a similarity pipeline run."""

    use_propagated_terms: bool
    ic_threshold: float | None
    top_k: int
    use_canonical_profiles: bool = True

    def to_dict(self) -> dict:
        """Return the run configuration as a JSON-serializable dictionary."""
        return {
            "use_propagated_terms": self.use_propagated_terms,
            "ic_threshold": self.ic_threshold,
            "top_k": self.top_k,
            "use_canonical_profiles": self.use_canonical_profiles,
        }


@dataclass
class PipelineConfig:
    """
    Configuration for running similarity pipelines.
    Maps to the RunConfig schema for embedding in MethodResults.
    """

    top_k: int = 10
    use_propagated_terms: bool = True
    ic_threshold: float = 1.5
    use_canonical_profiles: bool = True

    @property
    def terms_key(self) -> str:
        """Helper to determine which HPO term set to use based on config."""
        return "propagated_hpo_terms" if self.use_propagated_terms else "hpo_terms"

    def to_run_config(self) -> RunConfig:
        """Convert to RunConfig for embedding in MethodResults."""
        return RunConfig(
            use_propagated_terms=self.use_propagated_terms,
            ic_threshold=self.ic_threshold,
            top_k=self.top_k,
            use_canonical_profiles=self.use_canonical_profiles,
        )


@dataclass
class RunStats:
    """Runtime statistics observed during a similarity pipeline run."""

    n_patient_terms_raw: int
    n_patient_terms_propagated: int
    n_patient_terms_used: int
    n_diseases_scored: int
    n_diseases_skipped: int
    computation_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Return the run statistics as a JSON-serializable dictionary."""
        return {
            "n_patient_terms_raw": self.n_patient_terms_raw,
            "n_patient_terms_propagated": self.n_patient_terms_propagated,
            "n_patient_terms_used": self.n_patient_terms_used,
            "n_diseases_scored": self.n_diseases_scored,
            "n_diseases_skipped": self.n_diseases_skipped,
            "computation_time_seconds": round(self.computation_time_seconds, 4),
        }


@dataclass
class SimilarityResult:  # pylint: disable=too-many-instance-attributes
    """One ranked disease returned by a similarity method."""

    disease_id: str
    label: str
    score: float
    method_name: str
    profile_type: str | None = None
    category_source_id: str | None = None
    category_path: list[dict[str, object]] = field(default_factory=list)
    matched_aliases: list[str] = field(default_factory=list)
    rank: int = 0
    explanation: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the similarity result as a JSON-serializable dictionary."""
        return {
            "rank": self.rank,
            "disease_id": self.disease_id,
            "label": self.label,
            "profile_type": self.profile_type,
            "category_source_id": self.category_source_id,
            "category_path": self.category_path,
            "matched_aliases": self.matched_aliases,
            "score": round(self.score, 6),
            "method_name": self.method_name,
            "explanation": self.explanation,
        }


@dataclass
class MethodResults:
    """Complete output produced by one similarity method."""

    method_name: str
    pipeline_name: str
    config: RunConfig
    stats: RunStats
    rankings: list[SimilarityResult]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Return the full method output as a JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "method_name": self.method_name,
            "pipeline_name": self.pipeline_name,
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "rankings": [result.to_dict() for result in self.rankings],
        }
