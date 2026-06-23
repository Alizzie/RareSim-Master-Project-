"""
LLM retriever for direct rare disease retrieval and LLM-based explanations.

This file keeps the high-level orchestration out of methods.py.

methods.py:
    low-level HuggingFace loading, prompts, parsing, helper functions

retriever.py:
    high-level object that runs direct LLM retrieval and explains candidate results
"""

from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.llm.config import (
    EXPLAINER_MODEL,
    MAX_NEW_TOKENS_EXPLAINER,
    MAX_NEW_TOKENS_RETRIEVAL,
    TOP_K,
    TOP_K_RERANK,
)
from raresim.similarity_methods.llm.explanation import (
    build_explanation,
    build_metadata,
)
from raresim.similarity_methods.llm.methods import (
    as_string_list,
    build_disease_text_preview,
    build_explanation_prompt,
    build_patient_context_text,
    build_retrieval_prompt,
    extract_explanation,
    get_result_disease_id,
    load_hf_pipeline,
    parse_retrieval_output,
    query_hf,
    unload_pipeline,
)
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
        patient: dict,
        hpo_labels: dict[str, str],
        disease_profiles: dict[str, dict],
        *,
        disease_ancestors: dict[str, list[str]] | None = None,
        disease_metadata_index: dict[str, dict] | None = None,
    ) -> None:
        self.patient = patient
        self.hpo_labels = hpo_labels
        self.disease_profiles = disease_profiles
        self.disease_ancestors = disease_ancestors or {}
        self.disease_metadata_index = disease_metadata_index or {}

    def retrieve(
        self,
        model_name: str,
        top_k: int = TOP_K,
    ) -> tuple[list[dict], object]:
        """
        Use an LLM to directly retrieve and rank rare diseases.

        Returns:
            (results, pipe). The caller should call unload_pipeline(pipe).
        """
        prompt = build_retrieval_prompt(
            patient=self.patient,
            hpo_labels=self.hpo_labels,
            top_k=top_k,
        )

        print(f"\n[llm] Retrieving diseases with: {model_name}")

        with timer(f"load {model_name}"):
            pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_RETRIEVAL)

        with timer(f"generate {model_name}"):
            generated = query_hf(
                prompt,
                pipe,
                max_tokens=MAX_NEW_TOKENS_RETRIEVAL,
            )

        print("\n--- RAW LLM OUTPUT ---")
        print(generated)
        print("--- END OUTPUT ---\n")

        results = parse_retrieval_output(
            generated_text=generated,
            patient=self.patient,
            hpo_labels=self.hpo_labels,
            disease_profiles=self.disease_profiles,
            model_name=model_name,
            disease_ancestors=self.disease_ancestors,
            disease_metadata_index=self.disease_metadata_index,
            top_k=top_k,
        )

        n_validated = sum(
            1 for result in results if result.get("validated_against_profiles")
        )

        print(
            f"[llm] Found {len(results)} diseases "
            f"({n_validated} validated against profiles)"
        )

        return results, pipe

    def explain_results(  # pylint: disable=too-many-locals
        self,
        candidate_results: list[dict],
        *,
        model_name: str = EXPLAINER_MODEL,
        top_k: int = TOP_K_RERANK,
    ) -> list[dict]:
        """
        Add structured LLM explanations to candidate results.

        This can explain direct LLM retrieval results or transformer top-K results.
        """
        patient_hpo_terms = as_string_list(self.patient.get("hpo_terms", []))
        patient_text = build_patient_context_text(self.patient, self.hpo_labels)

        print(f"\n[llm] Loading explainer: {model_name}")

        pipe = None
        explained = []

        try:
            with timer("load explainer"):
                pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_EXPLAINER)

            candidates = candidate_results[:top_k]

            for index, original_result in enumerate(candidates, start=1):
                result = dict(original_result)
                disease_id = get_result_disease_id(result)

                if not disease_id or disease_id not in self.disease_profiles:
                    result["llm_explanation_text"] = "Disease profile not found."
                    explained.append(result)
                    continue

                disease = self.disease_profiles[disease_id]
                label = str(disease.get("label") or disease_id)

                print(f"  [llm] {index}/{len(candidates)}: {label}")

                with timer(f"explain {label[:40]}"):
                    prompt = build_explanation_prompt(
                        patient=self.patient,
                        disease=disease,
                        hpo_labels=self.hpo_labels,
                        candidate_score=result.get("score"),
                        candidate_rank=result.get("rank"),
                    )
                    generated = query_hf(
                        prompt,
                        pipe,
                        max_tokens=MAX_NEW_TOKENS_EXPLAINER,
                    )
                    explanation_text = extract_explanation(generated)

                category_metadata = build_category_metadata(
                    disease_id=disease_id,
                    profile=disease,
                    disease_ancestors=self.disease_ancestors,
                    disease_metadata_index=self.disease_metadata_index,
                )

                existing_aliases = result.get("matched_aliases", [])
                if not isinstance(existing_aliases, list):
                    existing_aliases = []

                matched_aliases = sorted(
                    {
                        *[str(alias) for alias in existing_aliases if alias],
                        *category_metadata["matched_aliases"],
                    }
                )

                disease_hpo_terms = as_string_list(disease.get("hpo_terms", []))
                disease_text_preview = build_disease_text_preview(
                    disease_profile=disease,
                    fallback_label=label,
                    hpo_labels=self.hpo_labels,
                )
                score = float(result.get("score", 0.0))

                result["profile_type"] = (
                    result.get("profile_type")
                    or category_metadata["profile_type"]
                )
                result["category_source_id"] = (
                    result.get("category_source_id")
                    or category_metadata["category_source_id"]
                )
                result["category_path"] = (
                    result.get("category_path")
                    or category_metadata["category_path"]
                )
                result["matched_aliases"] = matched_aliases

                result["llm_explanation_text"] = explanation_text
                result["explainer_model"] = model_name
                result["llm_explanation"] = build_explanation(
                    score=score,
                    model_name=model_name,
                    patient_text=patient_text,
                    disease_text_preview=disease_text_preview,
                    patient_hpo_terms=patient_hpo_terms,
                    disease_hpo_terms=disease_hpo_terms,
                    hpo_labels=self.hpo_labels,
                    llm_response=explanation_text,
                    prompt_name="llm_clinical_explanation",
                )
                result["llm_explanation_metadata"] = build_metadata(
                    model_name=model_name,
                    top_k=top_k,
                    prompt_name="llm_clinical_explanation",
                )

                explained.append(result)

        finally:
            if pipe is not None:
                unload_pipeline(pipe)

        return explained
