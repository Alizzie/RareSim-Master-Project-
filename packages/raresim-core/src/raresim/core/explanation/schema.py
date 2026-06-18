"""
Shared explanation schema for all similarity pipelines.

These dataclasses define the common spine that every method's explanation
must populate. The merger step relies on this structure being identical
across all methods. Do not add method-specific fields here.

Extension point: ExplanationBlock.method_specific (plain dict, method-owned).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TermEntry:
    """
    A single HPO term with its human-readable label and IC value.
    Used wherever a term needs to be presented to a user.
    """

    id: str
    label: str
    ic: float

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "ic": round(self.ic, 4)}


@dataclass
class TermMatch:
    """
    A matched HPO term — present in both patient and disease sets.

    match_type:
        "direct"     — term appears in both raw (non-propagated) sets.
        "propagated" — term appears only after true-path propagation.

    match_score:
        For set-based methods this is always 1.0 (binary presence).
        For semantic methods this is the pairwise BMA score for this term.
    """

    id: str
    label: str
    ic: float
    match_type: str = "direct"  # "direct" | "propagated"
    match_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "ic": round(self.ic, 4),
            "match_type": self.match_type,
            "match_score": round(self.match_score, 4),
        }


@dataclass
class CoverageBlock:
    """
    Standardized coverage metrics shared by every method.

    patient_coverage  : fraction of patient terms matched in disease.
    disease_coverage  : fraction of disease terms matched in patient.
    direction_asymmetry: |patient_coverage - disease_coverage|, surfaces
                         cases where one side is much broader than the other.
    """

    patient_coverage: float
    disease_coverage: float
    n_patient_terms: int
    n_disease_terms: int
    n_matched_terms: int
    n_unmatched_patient_terms: int
    direction_asymmetry: float = 0.0

    def to_dict(self) -> dict:
        return {
            "patient_coverage": round(self.patient_coverage, 4),
            "disease_coverage": round(self.disease_coverage, 4),
            "direction_asymmetry": round(self.direction_asymmetry, 4),
            "n_patient_terms": self.n_patient_terms,
            "n_disease_terms": self.n_disease_terms,
            "n_matched_terms": self.n_matched_terms,
            "n_unmatched_patient_terms": self.n_unmatched_patient_terms,
        }


@dataclass
class ExplanationBlock:
    """
    The complete explanation for one (patient, disease) pair.

    Fields that are ALWAYS present (shared spine):
        summary               — human-readable one-liner, template-generated.
        coverage              — CoverageBlock instance.
        matched_terms         — enriched list of shared HPO terms.
        unmatched_patient_terms — patient terms not found in disease.

    Extension point:
        method_specific       — dict owned entirely by each method's
                                explanation.py; arbitrary structure allowed.
        diagnostics           — debug/QA data, not shown to end users.
    """

    summary: str
    coverage: CoverageBlock
    matched_terms: list[TermMatch]
    unmatched_patient_terms: list[TermEntry]
    method_specific: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "coverage": self.coverage.to_dict(),
            "matched_terms": [t.to_dict() for t in self.matched_terms],
            "unmatched_patient_terms": [
                t.to_dict() for t in self.unmatched_patient_terms
            ],
            "method_specific": self.method_specific,
            "diagnostics": self.diagnostics,
        }
