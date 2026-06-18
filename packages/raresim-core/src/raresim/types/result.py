"""
Unified result schema for all similarity pipelines.

Every method returns a MethodResults object containing:
    - RunConfig   : what settings were used (static, set before the run)
    - RunStats    : what was observed during the run (computed after)
    - SimilarityResult list : one entry per ranked disease

AppMetadata is separate — it describes the loaded data, not a pipeline run.

"""

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class AppMetadata:
    """
    Describes the state of the loaded data.
    Computed once when AppContext is created, shared across all pipeline runs.

    n_hpo_labels          : total HPO terms in the ontology label map.
    n_disease_profiles    : total disease profiles loaded.
    unfound_patient_terms : patient HPO terms not present in hpo_labels
                            (may indicate outdated IDs or typos).
    """

    n_hpo_labels: int
    n_disease_profiles: int
    unfound_patient_terms: list[str]

    def to_dict(self) -> dict:
        return {
            "n_hpo_labels": self.n_hpo_labels,
            "n_disease_profiles": self.n_disease_profiles,
            "unfound_patient_terms": self.unfound_patient_terms,
        }


@dataclass
class RunConfig:
    """
    The configuration that governed this pipeline run.
    All fields are set before execution — none are computed from results.

    use_propagated_terms : whether true-path ancestor propagation was applied.
    ic_threshold         : minimum IC for a term to be used in scoring.
                           None means no filtering was applied.
    top_k                : number of top-ranked diseases returned.
    use_canonical_profiles: whether canonical (ORPHA-keyed) profiles were used.
    """

    use_propagated_terms: bool
    ic_threshold: float | None
    top_k: int
    use_canonical_profiles: bool = True

    def to_dict(self) -> dict:
        return {
            "use_propagated_terms": self.use_propagated_terms,
            "ic_threshold": self.ic_threshold,
            "top_k": self.top_k,
            "use_canonical_profiles": self.use_canonical_profiles,
        }


@dataclass
class RunStats:
    """
    Observations made during a pipeline run.
    All fields are computed from the data, not set by the caller.

    n_patient_terms_raw         : patient terms before propagation.
    n_patient_terms_propagated  : patient terms after true-path propagation.
    n_patient_terms_used        : patient terms actually used for scoring
                                  (after IC filtering, if any).
    n_diseases_scored           : disease profiles that produced a result
                                  (non-empty term set after filtering).
    n_diseases_skipped          : profiles skipped due to empty term set.
    computation_time_seconds    : wall-clock time for the scoring loop.
    """

    n_patient_terms_raw: int
    n_patient_terms_propagated: int
    n_patient_terms_used: int
    n_diseases_scored: int
    n_diseases_skipped: int
    computation_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_patient_terms_raw": self.n_patient_terms_raw,
            "n_patient_terms_propagated": self.n_patient_terms_propagated,
            "n_patient_terms_used": self.n_patient_terms_used,
            "n_diseases_scored": self.n_diseases_scored,
            "n_diseases_skipped": self.n_diseases_skipped,
            "computation_time_seconds": round(self.computation_time_seconds, 4),
        }


@dataclass
class SimilarityResult:
    """One ranked disease for a given patient-method pair."""

    disease_id: str
    label: str
    score: float
    method_name: str
    rank: int = 0
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "disease_id": self.disease_id,
            "label": self.label,
            "score": round(self.score, 6),
            "method_name": self.method_name,
            "explanation": self.explanation,
        }


@dataclass
class MethodResults:
    """
    All results for one method in one run.

    schema_version : version string for forward compatibility when loading
                     cached results. Increment when the output shape changes.
    method_name    : e.g. "set_jaccard", "semantic_resnik_bma".
    pipeline_name  : e.g. "set_based", "semantic".
    config         : RunConfig — what was set.
    stats          : RunStats  — what was observed.
    rankings       : top-k SimilarityResult objects, sorted by score.
    """

    method_name: str
    pipeline_name: str
    config: RunConfig
    stats: RunStats
    rankings: list[SimilarityResult]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "method_name": self.method_name,
            "pipeline_name": self.pipeline_name,
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "rankings": [r.to_dict() for r in self.rankings],
        }
