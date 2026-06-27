"""
Explanation utilities for LLM-based disease retrieval.

LLM scores/rankings are generated from prompt-based semantic comparison.
These explanations are descriptive.
"""

from typing import Any
from raresim.core.explanation import build_base_explanation
from raresim.similarity_methods.llm.config import (
    TEXT_PREVIEW_MAX_LENGTH,
)


def build_method_specific_explanation_block(
    model_name: str,
    patient_text: str,
    disease_text_preview: str,
    matched_phenotypes: str | Any,
    confidence: str | None = None,
    llm_response: str | None = None,
    prompt_name: str | None = None,
    method_specific_extra: dict | None = None,
) -> dict:
    """Build the method-specific explanation block for an LLM result."""
    return {
        "llm_method": "prompt_based_ranking",
        "model_name": model_name,
        "prompt_name": prompt_name,
        "method": "llm_retrieval",
        "confidence": confidence,
        "uses_ic_values": False,
        "uses_dense_embeddings": False,
        "uses_direct_hpo_overlap_for_score": False,
        "matched_phenotypes": matched_phenotypes,
        "score_note": (
            "LLM-based ranking score or confidence score. The model compares the "
            "patient profile with disease profile information using prompt-based "
            "semantic reasoning by the confidence scoring."
        ),
        "interpretation_note": (
            "Coverage and matched-term fields are descriptive HPO overlap on raw "
            "terms, shown for readability. They do NOT drive the LLM score and are "
            "not comparable to the propagated-term coverage of semantic methods."
        ),
        "patient_text_preview": patient_text[:TEXT_PREVIEW_MAX_LENGTH],
        "disease_text_preview": disease_text_preview[:TEXT_PREVIEW_MAX_LENGTH],
        "llm_response_preview": (llm_response or "")[:TEXT_PREVIEW_MAX_LENGTH],
        **(method_specific_extra or {}),
    }


def build_explanation(  # pylint: disable=too-many-arguments
    *,
    score: float,
    model_name: str,
    patient_hpo_terms: list[str],
    disease_hpo_terms: list[str],
    ic_values: dict[str, float],
    hpo_labels: dict[str, str],
    method_specific: dict[str, Any] | None = None,
    diagnostics_extras: dict[str, Any] | None = None,
) -> dict:
    """Build a standardized explanation dictionary for one LLM disease result."""

    summary = (
        f"LLM semantic ranking ({model_name}) assigned score {score:.4f}. "
        "The score is based on prompt-based semantic comparison, not "
        "IC-based HPO scoring."
    )

    return build_base_explanation(
        patient_terms=set(patient_hpo_terms),
        disease_terms=set(disease_hpo_terms),
        hpo_labels=hpo_labels,
        ic_values=ic_values,
        summary=summary,
        method_specific=method_specific,
        diagnostics={"raw_score": round(score, 6), **(diagnostics_extras or {})},
    ).to_dict()
