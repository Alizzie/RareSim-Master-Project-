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
    app: AppMetadata

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "pipeline_name": self.pipeline_name,
            "use_propagated_terms": self.use_propagated_terms,
            "ic_threshold": self.ic_threshold,
            "top_k": self.top_k,
            "n_patient_terms": self.n_patient_terms,
            "n_disease_terms": self.n_disease_terms,
            "app": self.app.to_dict(),
        }


@dataclass
class SimilarityResult:
    disease_id: str
    label: str
    score: float
    method_name: str
    metadata: Metadata
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
            "metadata": self.metadata.to_dict(),
        }
