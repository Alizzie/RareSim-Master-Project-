"""
LLM retriever for direct rare disease retrieval and LLM-based explanations.

This file keeps the high-level orchestration out of methods.py.

methods.py:
    low-level HuggingFace loading, prompts, parsing, helper functions

retriever.py:
    high-level object that runs direct LLM retrieval and explains candidate results
"""

from typing import cast
from raresim.similarity_methods.llm.config import (
    EXPLAINER_MODEL,
    MAX_NEW_TOKENS_EXPLAINER,
    MAX_NEW_TOKENS_RETRIEVAL,
    TOP_K_RERANK,
    TEXT_PREVIEW_MAX_LENGTH,
)
from raresim.core import build_run_stats, AppContext
from raresim.similarity_methods.llm.methods import (
    build_explanation_prompt,
    build_patient_context_text,
    build_retrieval_prompt,
    parse_explanation,
    parse_retrieval_output,
    query_hf,
)
from raresim.types import RunStats, SimilarityResult, PatientProfile
from raresim.utils.timer import timer


class LlmDiseaseRetriever:
    """
    High-level LLM retrieval/explanation object.

    It can:
    - directly ask an LLM to retrieve disease candidates
    - explain already-ranked candidates, including direct LLM or transformer results
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        patient: PatientProfile,
        hpo_labels: dict[str, str],
        disease_profiles: dict[str, dict],
        *,
        ic_values: dict[str, float] | None = None,
        disease_ancestors: dict[str, list[str]] | None = None,
        disease_metadata_index: dict[str, dict] | None = None,
    ) -> None:
        self.patient = patient
        self.hpo_labels = hpo_labels
        self.ic_values = ic_values or {}
        self.disease_profiles = disease_profiles
        self.disease_ancestors = disease_ancestors or {}
        self.disease_metadata_index = disease_metadata_index or {}

    @classmethod
    def from_context(
        cls, patient: PatientProfile, context: AppContext
    ) -> "LlmDiseaseRetriever":
        """
        Create a retriever from a patient and an AppContext.

        This is a convenience method to avoid passing many arguments.
        """
        return cls(
            patient=patient,
            hpo_labels=context.hpo_labels,
            disease_profiles=context.disease_profiles,
            ic_values=context.ic_values,
            disease_ancestors=context.disease_ancestors,
            disease_metadata_index=context.disease_metadata_index,
        )


    def retrieve_with_pipe(
        self,
        pipe,
        model_name: str,
        top_k: int,
    ) -> list[SimilarityResult]:
        """Uses an already-loaded pipeline."""
        prompt = build_retrieval_prompt(
            patient=self.patient,
            hpo_labels=self.hpo_labels,
            top_k=top_k,
        )

        print(f"\n[llm] Retrieving diseases with: {model_name}")

        with timer(f"generate {model_name}"):
            generated = query_hf(prompt, pipe, max_tokens=MAX_NEW_TOKENS_RETRIEVAL)

        print("\n--- RAW LLM OUTPUT ---")
        print(generated)
        print("--- END OUTPUT ---\n")

        rankings = parse_retrieval_output(
            generated_text=generated,
            patient=self.patient,
            hpo_labels=self.hpo_labels,
            ic_values=self.ic_values,
            disease_profiles=self.disease_profiles,
            model_name=model_name,
            disease_ancestors=self.disease_ancestors,
            disease_metadata_index=self.disease_metadata_index,
            top_k=top_k,
        )

        n_validated = sum(
            1
            for r in rankings
            if cast(dict, r.explanation.get("diagnostics", {})).get(
                "validated_against_profiles"
            )
        )
        print(f"[llm] Found {len(rankings)} diseases ({n_validated} validated against profiles)")

        return rankings


    def explain_results_with_pipe(  # pylint: disable=too-many-locals
        self,
        pipe,
        candidate_results: list[SimilarityResult],
        *,
        model_name: str = EXPLAINER_MODEL,
        top_k: int = TOP_K_RERANK,
    ) -> list[SimilarityResult]:
        """Explain candidate results using an already-loaded pipeline."""
        patient_text = build_patient_context_text(self.patient, self.hpo_labels)
        candidates = candidate_results[:top_k]

        # Build all prompts up front, skipping candidates with no disease profile.
        prompts = []
        prompt_targets = []  # parallel list of (result, method_specific) for each prompt
        explained: list[SimilarityResult] = []

        for result in candidates:
            disease_id = result.disease_id
            method_specific = cast(dict, result.explanation.setdefault("method_specific", {}))

            if not disease_id or disease_id not in self.disease_profiles:
                method_specific["clinical_explanation"] = "Disease profile not found."
                explained.append(result)
                continue

            disease = self.disease_profiles[disease_id]
            prompts.append(
                build_explanation_prompt(
                    patient=self.patient,
                    disease=disease,
                    hpo_labels=self.hpo_labels,
                    candidate_score=result.score,
                    candidate_rank=result.rank,
                )
            )
            prompt_targets.append((result, method_specific))

        if prompts:
            with timer(f"explain batch of {len(prompts)}"):
                outputs = pipe(prompts, max_new_tokens=MAX_NEW_TOKENS_EXPLAINER)

            for (result, method_specific), output, prompt in zip(prompt_targets, outputs, prompts):
                generated_text = output[0]["generated_text"]
                generated = generated_text[len(prompt):].strip() if generated_text else ""
                explanation_text = parse_explanation(generated)

                method_specific["clinical_explanation"] = explanation_text["text"]
                method_specific["verdict"] = explanation_text.get("verdict", None)
                method_specific["verdict_reason"] = explanation_text.get("verdict_reason", None)
                method_specific["explainer_model"] = model_name
                method_specific["patient_text_preview"] = patient_text[:TEXT_PREVIEW_MAX_LENGTH]

                explained.append(result)

        return explained


    def run_stats(self, rankings: list[SimilarityResult], elapsed: float) -> RunStats:
        """Build run stats for LLM retrieval."""
        n_patient_terms = len(self.patient.get_terms(use_propagated=False))
        return build_run_stats(
            n_patient_terms_raw=n_patient_terms,
            n_patient_terms_propagated=0,
            n_patient_terms_used=n_patient_terms,
            n_diseases_scored=len(rankings),
            n_diseases_skipped=0,
            computation_time=elapsed,
        )
