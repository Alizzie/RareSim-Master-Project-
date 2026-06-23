"""
LLM methods — HuggingFace backend, prompt builders, parsers, and helpers.

High-level orchestration should live in retriever.py.
This module keeps reusable low-level utilities and legacy wrapper functions.
"""

import gc
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.llm.config import (
    DO_SAMPLE,
    EXPLAINER_MODEL,
    MAX_NEW_TOKENS_EXPLAINER,
    MAX_NEW_TOKENS_RETRIEVAL,
    TEMPERATURE,
    TOP_K,
    TOP_K_RERANK,
)
from raresim.similarity_methods.llm.explanation import (
    build_explanation,
    build_metadata,
)
from raresim.utils.timer import timer


def as_string_list(value: object) -> list[str]:
    """Convert unknown list-like input into a list of strings."""
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    return []


def merge_aliases(*alias_groups: object) -> list[str]:
    """Merge alias groups into a sorted unique alias list."""
    aliases = set()

    for group in alias_groups:
        if isinstance(group, str):
            aliases.add(group)
        elif isinstance(group, (list, tuple, set)):
            aliases.update(str(alias) for alias in group if alias)

    return sorted(aliases)


def get_hpo_label(term: str, hpo_labels: dict[str, str]) -> str:
    """Return the HPO label for a term, falling back to the term ID."""
    return hpo_labels.get(term) or term


# ── HuggingFace backend ───────────────────────────────────────────────────────


def load_hf_pipeline(model_name: str, max_new_tokens: int = MAX_NEW_TOKENS_RETRIEVAL):
    """
    Load a HuggingFace text-generation pipeline.

    Tries 4-bit quantization first and falls back to float16 if quantized
    loading is not available.
    """
    print(f"  [llm] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        print("  [llm] Loaded with 4-bit quantization")
    except (ImportError, OSError, RuntimeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("  [llm] Loaded with float16 precision")

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        repetition_penalty=1.3,
    )


def unload_pipeline(pipe) -> None:
    """
    Release GPU memory after a model is done.

    Deletes the pipeline, runs garbage collection, and clears CUDA cache.
    """
    del pipe
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  [llm] GPU memory released")


def query_hf(
    prompt: str,
    pipe,
    max_tokens: int = MAX_NEW_TOKENS_RETRIEVAL,
) -> str:
    """
    Query a loaded HuggingFace pipeline.

    Strips the input prompt from the output and returns only generated text.
    """
    output = pipe(prompt, max_new_tokens=max_tokens)
    generated = output[0]["generated_text"]
    return generated[len(prompt) :].strip() if generated else ""


# ── Shared text/result helpers ────────────────────────────────────────────────


def confidence_to_score(confidence: str | None) -> float:
    """Convert textual LLM confidence into a numeric score for ranking displays."""
    value = (confidence or "").strip().lower()

    if value.startswith("high") or "strong" in value:
        return 0.9
    if value.startswith("medium") or "possible" in value:
        return 0.6
    if value.startswith("low") or "weak" in value:
        return 0.3

    return 0.5


def build_patient_context_text(patient: dict, hpo_labels: dict[str, str]) -> str:
    """Build the patient text shown in LLM explanations."""
    raw_text = str(patient.get("raw_text") or "").strip()
    hpo_terms = as_string_list(patient.get("hpo_terms", []))
    hpo_term_labels = [get_hpo_label(term, hpo_labels) for term in hpo_terms]

    parts = []
    if raw_text:
        parts.append(f"Clinical description: {raw_text}")
    if hpo_term_labels:
        parts.append(f"Patient phenotypes: {', '.join(hpo_term_labels)}")

    return "\n".join(parts).strip()


def build_disease_text_preview(
    disease_profile: dict,
    fallback_label: str,
    hpo_labels: dict[str, str],
) -> str:
    """Build a short disease profile preview for LLM explanations."""
    label = disease_profile.get("label") or fallback_label
    description = str(disease_profile.get("merged_description") or "").strip()

    hpo_terms = as_string_list(disease_profile.get("hpo_terms", []))
    hpo_term_labels = [get_hpo_label(term, hpo_labels) for term in hpo_terms[:20]]

    parts = [f"Disease: {label}"]

    if description:
        parts.append(f"Description: {description[:400]}")

    if hpo_term_labels:
        parts.append(f"Phenotypes: {', '.join(hpo_term_labels)}")

    return "\n".join(parts)


def get_result_disease_id(result: dict) -> str | None:
    """Extract a disease ID from heterogeneous result dictionaries."""
    return (
        result.get("canonical_disease_id")
        or result.get("disease_id")
        or result.get("ordo_id")
    )


# ── Retrieval prompt builder ──────────────────────────────────────────────────


def build_retrieval_prompt(
    patient: dict,
    hpo_labels: dict[str, str],
    top_k: int = TOP_K,
) -> str:
    """Build the prompt that asks the LLM to directly retrieve diseases."""
    hpo_terms = as_string_list(patient.get("hpo_terms", []))
    hpo_term_labels = [get_hpo_label(term, hpo_labels) for term in hpo_terms]
    raw_text = str(patient.get("raw_text") or "").strip()

    content_parts = [
        "You are a rare disease expert specializing in clinical phenotyping.",
    ]

    if raw_text:
        content_parts.append(f"Clinical description: {raw_text}")

    content_parts += [
        f"Patient phenotypes: {', '.join(hpo_term_labels)}",
        "",
        f"You MUST list exactly {top_k} DIFFERENT rare diseases.",
        "Do not repeat the same disease.",
        f"List the top {top_k} most likely rare diseases for this patient.",
        "Format each result on a SINGLE LINE exactly like this:",
        (
            "DISEASE: <name> | ORDO: ORPHA:<number> | MATCH: <phenotypes> | "
            "CONFIDENCE: <high/medium/low>"
        ),
        "",
        "Example:",
        (
            "DISEASE: Friedreich Ataxia | ORDO: ORPHA:95 | "
            "MATCH: cerebellar ataxia, anemia | CONFIDENCE: high"
        ),
        (
            "DISEASE: Wilson Disease | ORDO: ORPHA:905 | "
            "MATCH: developmental delay, anemia | CONFIDENCE: medium"
        ),
        (
            "DISEASE: Gaucher Disease | ORDO: ORPHA:355 | "
            "MATCH: anemia, developmental delay | CONFIDENCE: low"
        ),
        "",
        f"Now list {top_k} different diseases:",
    ]

    content = "\n".join(content_parts)
    return f"[INST] {content} [/INST]"


# ── Retrieval output parser ───────────────────────────────────────────────────


def find_disease_in_profiles(
    ordo_id: str,
    disease_name: str,
    disease_profiles: dict[str, dict],
) -> tuple[str, str, bool]:
    """
    Find a disease in profiles by ORPHA ID first, then fuzzy name match.

    Returns:
        (matched_id, label, validated) where validated=True if found in profiles.
    """
    if ordo_id in disease_profiles:
        return ordo_id, disease_profiles[ordo_id].get("label", disease_name), True

    normalized_id = ordo_id.replace(":", "")
    for profile_id in disease_profiles:
        if profile_id.replace(":", "") == normalized_id:
            label = disease_profiles[profile_id].get("label", disease_name)
            return profile_id, label, True

    disease_name_lower = disease_name.lower()
    for profile_id, profile in disease_profiles.items():
        label = profile.get("label", "").lower()
        if disease_name_lower in label or label in disease_name_lower:
            return profile_id, profile.get("label", disease_name), True

    return ordo_id, disease_name, False


def parse_retrieval_output(  # pylint: disable=too-many-arguments,too-many-locals
    generated_text: str,
    patient: dict,
    hpo_labels: dict[str, str],
    disease_profiles: dict[str, dict],
    model_name: str,
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Parse LLM generated text into structured disease results.

    Handles common LLM formatting variations, validates each result against the
    disease profiles, and enriches validated diseases with profile/category
    metadata, explanation, and run metadata.
    """
    if not generated_text:
        return []

    disease_ancestors = disease_ancestors or {}
    disease_metadata_index = disease_metadata_index or {}
    patient_hpo_terms = as_string_list(patient.get("hpo_terms", []))
    patient_text = build_patient_context_text(patient, hpo_labels)

    generated_text = re.sub(r"ORPHA(\d+)", r"ORPHA:\1", generated_text)
    generated_text = re.sub(r"OMIM(\d+)", r"OMIM:\1", generated_text)
    generated_text = re.sub(r"\[SOLUTION\]", "", generated_text)
    generated_text = re.sub(
        r"\[INST\].*?\[/INST\]",
        "",
        generated_text,
        flags=re.DOTALL,
    )

    normalized = " ".join(
        line.strip() for line in generated_text.splitlines() if line.strip()
    )

    pattern = re.compile(
        r"DISEASE:\s*(.+?)\s*\|\s*ORDO:\s*([\w:]+)\s*\|\s*MATCH:\s*(.+?)"
        r"\s*\|\s*\w+:\s*([\w\s]+?)(?=DISEASE:|$)",
        re.IGNORECASE,
    )

    results = []
    rank = 1
    seen_ids = set()

    for match in pattern.finditer(normalized):
        disease_name = match.group(1).strip()
        ordo_id = match.group(2).strip()
        matched_phenotypes = match.group(3).strip()
        confidence = match.group(4).strip().lower()

        matched_id, label, validated = find_disease_in_profiles(
            ordo_id,
            disease_name,
            disease_profiles,
        )

        if matched_id in seen_ids:
            continue
        seen_ids.add(matched_id)

        disease_profile = disease_profiles.get(matched_id, {})
        disease_hpo_terms = as_string_list(disease_profile.get("hpo_terms", []))
        score = confidence_to_score(confidence)

        category_metadata = build_category_metadata(
            disease_id=matched_id,
            profile=disease_profile,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
        )

        disease_text_preview = build_disease_text_preview(
            disease_profile=disease_profile,
            fallback_label=label,
            hpo_labels=hpo_labels,
        )

        results.append(
            {
                "rank": rank,
                "canonical_disease_id": matched_id,
                "representative_disease_id": matched_id,
                "disease_id": matched_id,
                "ordo_id": matched_id,
                "label": label,
                "disease_name": label,
                "profile_type": category_metadata["profile_type"],
                "category_source_id": category_metadata["category_source_id"],
                "category_path": category_metadata["category_path"],
                "matched_aliases": category_metadata["matched_aliases"],
                "score": score,
                "matched_phenotypes": matched_phenotypes,
                "confidence": confidence,
                "validated_against_profiles": validated,
                "model_name": model_name,
                "model": model_name,
                "method": "llm_retrieval",
                "explanation": build_explanation(
                    score=score,
                    model_name=model_name,
                    patient_text=patient_text,
                    disease_text_preview=disease_text_preview,
                    patient_hpo_terms=patient_hpo_terms,
                    disease_hpo_terms=disease_hpo_terms,
                    hpo_labels=hpo_labels,
                    llm_response=generated_text,
                    prompt_name="llm_direct_retrieval",
                ),
                "metadata": build_metadata(
                    model_name=model_name,
                    top_k=top_k,
                    prompt_name="llm_direct_retrieval",
                ),
            }
        )

        rank += 1
        if rank > top_k:
            break

    return results


# ── Retrieval entry point ─────────────────────────────────────────────────────


def retrieve_diseases_llm(  # pylint: disable=too-many-arguments
    patient: dict,
    hpo_labels: dict[str, str],
    disease_profiles: dict[str, dict],
    model_name: str,
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    top_k: int = TOP_K,
) -> tuple[list[dict], object]:
    """
    Use an LLM to directly retrieve and rank rare diseases from patient HPO terms.

    Loads the model, generates results, then returns both the results and the
    pipeline object so the caller can unload it when ready.
    """
    prompt = build_retrieval_prompt(patient, hpo_labels, top_k)

    print(f"\n[llm] Retrieving diseases with: {model_name}")

    with timer(f"load {model_name}"):
        pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_RETRIEVAL)

    with timer(f"generate {model_name}"):
        generated = query_hf(prompt, pipe, max_tokens=MAX_NEW_TOKENS_RETRIEVAL)

    print("\n--- RAW LLM OUTPUT ---")
    print(generated)
    print("--- END OUTPUT ---\n")

    results = parse_retrieval_output(
        generated_text=generated,
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
        model_name=model_name,
        disease_ancestors=disease_ancestors,
        disease_metadata_index=disease_metadata_index,
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


# ── Explanation prompt builder ────────────────────────────────────────────────


def build_explanation_prompt(  # pylint: disable=too-many-locals
    patient: dict,
    disease: dict,
    hpo_labels: dict[str, str],
    candidate_score: float | None = None,
    candidate_rank: int | None = None,
) -> str:
    """Build a structured clinical reasoning prompt for one patient-disease pair."""
    patient_terms = as_string_list(patient.get("hpo_terms", []))
    disease_terms = as_string_list(disease.get("hpo_terms", []))

    patient_labels = [get_hpo_label(term, hpo_labels) for term in patient_terms]
    disease_labels = [get_hpo_label(term, hpo_labels) for term in disease_terms]

    disease_label = str(disease.get("label") or "Unknown disease")
    disease_desc = str(disease.get("merged_description") or "").strip()

    patient_set = set(patient_terms)
    disease_set = set(disease_terms)
    matching_ids = patient_set & disease_set
    matching_labels = [get_hpo_label(term, hpo_labels) for term in matching_ids]
    missing_from_disease = [
        get_hpo_label(term, hpo_labels) for term in patient_set - disease_set
    ]

    prompt_parts = [
        "You are a rare disease clinical expert. Your task is to evaluate whether "
        "a patient's phenotype profile matches a candidate rare disease.",
        "",
        "---",
        (
            "PATIENT PHENOTYPES: "
            f"{', '.join(patient_labels) if patient_labels else 'None listed'}"
        ),
        "---",
        f"CANDIDATE DISEASE: {disease_label}",
    ]

    if disease_desc:
        prompt_parts.append(f"DISEASE DESCRIPTION: {disease_desc[:400]}")

    if disease_labels:
        prompt_parts.append(f"DISEASE PHENOTYPES: {', '.join(disease_labels[:20])}")

    if candidate_score is not None:
        prompt_parts += [
            "",
            f"Candidate similarity/ranking score: {candidate_score:.4f} "
            f"(rank #{candidate_rank})",
        ]

    prompt_parts += [
        "",
        "Pre-computed phenotype overlap for your reference:",
        (
            f"  Directly matched: {', '.join(matching_labels)}"
            if matching_labels
            else "  Directly matched: None"
        ),
        (
            f"  In patient but not in disease: {', '.join(missing_from_disease)}"
            if missing_from_disease
            else "  In patient but not in disease: None"
        ),
        "",
        "Provide a structured clinical assessment using EXACTLY this format:",
        "",
        "CLINICAL REASONING: [2-3 sentences explaining which phenotypes match, "
        "which are missing, and any clinically important discrepancies.]",
        "",
        "VERDICT: [STRONG MATCH / POSSIBLE MATCH / WEAK MATCH / UNLIKELY MATCH]",
        "",
        "VERDICT REASON: [One sentence justifying the verdict.]",
        "",
        "CLINICAL REASONING:",
    ]

    return "\n".join(prompt_parts)


# ── Explanation extractor ─────────────────────────────────────────────────────


def extract_explanation(generated_text: str) -> str:  # pylint: disable=too-many-branches
    """Extract and structure the explanation from LLM output."""
    if not generated_text:
        return "No explanation generated."

    generated_text = generated_text.strip()
    reasoning = ""
    verdict = ""
    verdict_reason = ""
    current_section = None

    for line in generated_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.upper().startswith("VERDICT REASON:"):
            current_section = "verdict_reason"
            verdict_reason = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("VERDICT:"):
            current_section = "verdict"
            verdict = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("CLINICAL REASONING:"):
            current_section = "reasoning"
            after = line.split(":", 1)[-1].strip()
            if after:
                reasoning = after
        elif current_section == "reasoning" and not verdict:
            reasoning += " " + line
        elif current_section == "verdict" and not verdict_reason:
            if not line.upper().startswith("VERDICT"):
                verdict += " " + line

    if verdict:
        parts = []
        if reasoning:
            parts.append(reasoning.strip())
        parts.append(f"Verdict: {verdict.strip()}")
        if verdict_reason:
            parts.append(verdict_reason.strip())
        return " | ".join(parts)

    sentences = generated_text.split(".")
    short = ". ".join(sentence.strip() for sentence in sentences[:3] if sentence.strip())
    return short + "." if short and not short.endswith(".") else short


# ── Explanation entry point ───────────────────────────────────────────────────


def explain_top_results(  # pylint: disable=too-many-arguments,too-many-locals
    patient: dict,
    candidate_results: list[dict],
    disease_profiles: dict[str, dict],
    hpo_labels: dict[str, str],
    *,
    disease_ancestors: dict[str, list[str]] | None = None,
    disease_metadata_index: dict[str, dict] | None = None,
    model_name: str = EXPLAINER_MODEL,
    top_k: int = TOP_K_RERANK,
) -> list[dict]:
    """
    Add structured LLM explanations to top-K candidate results.

    This can explain direct LLM retrieval results or transformer top-K results.
    """
    disease_ancestors = disease_ancestors or {}
    disease_metadata_index = disease_metadata_index or {}
    patient_hpo_terms = as_string_list(patient.get("hpo_terms", []))
    patient_text = build_patient_context_text(patient, hpo_labels)

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

            if not disease_id or disease_id not in disease_profiles:
                result["llm_explanation_text"] = "Disease profile not found."
                explained.append(result)
                continue

            disease = disease_profiles[disease_id]
            label = str(disease.get("label") or disease_id)

            print(f"  [llm] {index}/{len(candidates)}: {label}")

            with timer(f"explain {label[:40]}"):
                prompt = build_explanation_prompt(
                    patient=patient,
                    disease=disease,
                    hpo_labels=hpo_labels,
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
                disease_ancestors=disease_ancestors,
                disease_metadata_index=disease_metadata_index,
            )

            disease_hpo_terms = as_string_list(disease.get("hpo_terms", []))
            disease_text_preview = build_disease_text_preview(
                disease_profile=disease,
                fallback_label=label,
                hpo_labels=hpo_labels,
            )
            score = float(result.get("score", 0.0))

            result["profile_type"] = (
                result.get("profile_type") or category_metadata["profile_type"]
            )
            result["category_source_id"] = (
                result.get("category_source_id")
                or category_metadata["category_source_id"]
            )
            result["category_path"] = (
                result.get("category_path") or category_metadata["category_path"]
            )
            result["matched_aliases"] = merge_aliases(
                result.get("matched_aliases", []),
                category_metadata["matched_aliases"],
            )

            result["llm_explanation_text"] = explanation_text
            result["explainer_model"] = model_name
            result["llm_explanation"] = build_explanation(
                score=score,
                model_name=model_name,
                patient_text=patient_text,
                disease_text_preview=disease_text_preview,
                patient_hpo_terms=patient_hpo_terms,
                disease_hpo_terms=disease_hpo_terms,
                hpo_labels=hpo_labels,
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
