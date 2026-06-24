"""
Shared explanation schema for all similarity pipelines.

These dataclasses define the common spine that every method's explanation
must populate. The merger step relies on this structure being identical
across all methods. Do not add method-specific fields here.

Extension point: ExplanationBlock.method_specific (plain dict, method-owned).
"""

from dataclasses import dataclass, field
from typing import Any, Literal


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
class TokenEntry:
    """
    A single clinical text token with its IDF weight.
    """

    token: str  # e.g. "cerebellar"
    idf_weight: float  # IDF value from the text corpus

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "idf_weight": round(self.idf_weight, 4),
        }


@dataclass
class TokenMatch:
    """
    A matched token, present in both patient and disease token vectors.
    Used for matched tokens in text and hybrid TF-IDF modes.
    """

    token: str
    idf_weight: float
    match_score: float = 1.0

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "idf_weight": round(self.idf_weight, 4),
            "match_score": round(self.match_score, 4),
        }


@dataclass
class HpoCoverageBlock:
    """
    Coverage metrics when patient and disease are compared as HPO term sets.

    Used by: set-based, semantic, TF-IDF HPO mode, HPO2Vec, autoencoder.

    patient_coverage    : fraction of patient HPO terms matched in disease.
    disease_coverage    : fraction of disease HPO terms matched in patient.
    direction_asymmetry : |patient_coverage - disease_coverage|. A large
                          value means one side is much broader than the other.
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
            "patient_hpo_coverage": round(self.patient_coverage, 4),
            "disease_hpo_coverage": round(self.disease_coverage, 4),
            "direction_asymmetry": round(self.direction_asymmetry, 4),
            "n_patient_terms": self.n_patient_terms,
            "n_disease_terms": self.n_disease_terms,
            "n_matched_terms": self.n_matched_terms,
            "n_unmatched_patient_terms": self.n_unmatched_patient_terms,
        }


@dataclass
class TokenCoverageBlock:
    """
    Coverage metrics when patient and disease are compared as token sets.

    Used by: TF-IDF text mode, TF-IDF hybrid mode.

    Field names use *_tokens suffix throughout to avoid any confusion with
    HpoCoverageBlock's *_terms fields — the units are different and should
    never be silently swapped.

    sparse_disease_description : True when the disease has fewer than 10
                                 tokens. Scores against very short descriptions
                                 are unreliable (matching 2/2 tokens gives a
                                 perfect disease_token_coverage regardless of
                                 clinical relevance).
    """

    patient_token_coverage: float
    disease_token_coverage: float
    n_patient_tokens: int
    n_disease_tokens: int
    n_matched_tokens: int
    n_unmatched_patient_tokens: int
    direction_asymmetry: float = 0.0
    is_sparse_disease: bool = False

    def to_dict(self) -> dict:
        return {
            "patient_token_coverage": round(self.patient_token_coverage, 4),
            "disease_token_coverage": round(self.disease_token_coverage, 4),
            "direction_asymmetry": round(self.direction_asymmetry, 4),
            "n_patient_tokens": self.n_patient_tokens,
            "n_disease_tokens": self.n_disease_tokens,
            "n_matched_tokens": self.n_matched_tokens,
            "n_unmatched_patient_tokens": self.n_unmatched_patient_tokens,
            "sparse_disease_description": self.is_sparse_disease,
        }


CoverageBlock = HpoCoverageBlock | TokenCoverageBlock
MatchedTerm = TermMatch | TokenMatch
UnmatchedTerm = TermEntry | TokenEntry


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
    matched_terms: list[TokenMatch] | list[TermMatch]
    unmatched_patient_terms: list[TokenEntry] | list[TermEntry]
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
