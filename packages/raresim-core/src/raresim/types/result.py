"""
Unified result schema for all similarity pipelines.
Every method returns a list of SimilarityResult, regardless of pipeline type.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppMetadata:
    """
    One-time summary computed when the app loads.
    Captures the state of the data, not the method configuration.
    """

    n_hpo_labels: int
    n_disease_profiles: int
    n_patient_terms: int
    n_patient_propagated_terms: int
    unfound_patient_terms: list[str]

    def to_dict(self) -> dict:
        return {
            "n_hpo_labels": self.n_hpo_labels,
            "n_disease_profiles": self.n_disease_profiles,
            "n_patient_terms": self.n_patient_terms,
            "n_patient_propagated_terms": self.n_patient_propagated_terms,
            "unfound_patient_terms": self.unfound_patient_terms,
        }


@dataclass
class Metadata:
    """
    Per-method configuration metadata.
    Pipeline-specific diagnostics belong in SimilarityResult.explanation.
    """

    method_name: str
    pipeline_name: str
    use_propagated_terms: bool
    ic_threshold: float | None
    top_k: int
    n_patient_terms: int
    n_disease_terms: int
    computation_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "pipeline_name": self.pipeline_name,
            "use_propagated_terms": self.use_propagated_terms,
            "ic_threshold": self.ic_threshold,
            "top_k": self.top_k,
            "n_used_patient_terms": self.n_patient_terms,
            "n_used_disease_terms": self.n_disease_terms,
            "computation_time": self.computation_time,
        }


@dataclass
class SimilarityResult:
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
            "method_name": self.method_name,
            "score": self.score,
            "explanation": self.explanation,
        }


@dataclass
class MethodResults:
    """Groups ranked results and shared metadata for one method"""

    metadata: Metadata
    rankings: list[SimilarityResult]

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "rankings": [
                {
                    "rank": r.rank,
                    "disease_id": r.disease_id,
                    "label": r.label,
                    "score": r.score,
                    "explanation": r.explanation,
                }
                for r in self.rankings
            ],
        }
