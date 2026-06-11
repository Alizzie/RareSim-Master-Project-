"""
LLM methods — disease retrieval and explanation via HuggingFace.

Two main functions:
- retrieve_diseases_llm : directly asks LLM to name diseases from patient HPO terms
- explain_top_results   : asks LLM to explain why each disease matches the patient
"""

import gc
import re
from typing import Dict, List, Tuple

from raresim.utils.timer import timer
from raresim.similarity_methods.llm.config import (
    DO_SAMPLE,
    EXPLAINER_MODEL,
    MAX_NEW_TOKENS_EXPLAINER,
    MAX_NEW_TOKENS_RETRIEVAL,
    TEMPERATURE,
    TOP_K,
    TOP_K_RERANK,
)

# ── HuggingFace backend ───────────────────────────────────────────────────────


def load_hf_pipeline(model_name: str, max_new_tokens: int = MAX_NEW_TOKENS_RETRIEVAL):
    """
    Load a HuggingFace text-generation pipeline.

    Tries 4-bit quantization first (less GPU memory) and falls back
    to float16 if quantization is not available.

    Args:
        model_name:     HuggingFace model identifier.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        HuggingFace text-generation pipeline.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        print(f"  [llm] Loading: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try 4-bit quantization first — uses less GPU memory
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
            print(f"  [llm] Loaded with 4-bit quantization")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print(f"  [llm] Loaded with float16 precision")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            repetition_penalty=1.3,
        )

    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Install with: pip install transformers torch"
        )


def unload_pipeline(pipe) -> None:
    """
    Release GPU memory after a model is done.

    Deletes the pipeline, runs garbage collection, and clears CUDA cache.
    Call this after each model finishes to free memory for the next model.
    """
    import torch

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("  [llm] GPU memory released")


def query_hf(
    prompt: str,
    pipe,
    max_tokens: int = MAX_NEW_TOKENS_RETRIEVAL,
) -> str:
    """
    Query a loaded HuggingFace pipeline.
    Strips the input prompt from the output — returns only generated text.
    """
    output = pipe(prompt, max_new_tokens=max_tokens)
    generated = output[0]["generated_text"]
    return generated[len(prompt) :].strip() if generated else ""


# ── Retrieval prompt builder ──────────────────────────────────────────────────


def build_retrieval_prompt(
    patient: dict,
    hpo_labels: Dict[str, str],
    top_k: int = TOP_K,
) -> str:
    hpo_term_labels = [hpo_labels.get(t, t) for t in patient.get("hpo_terms", [])]
    raw_text = patient.get("raw_text", "").strip()

    content_parts = [
        "You are a rare disease expert specializing in clinical phenotyping.",
    ]

    if raw_text:
        content_parts.append(f"Clinical description: {raw_text}")

    content_parts += [
        f"Patient phenotypes: {', '.join(hpo_term_labels)}",
        "",
        f"You MUST list exactly {top_k} DIFFERENT rare diseases. Do not repeat the same disease.",
        f"List the top {top_k} most likely rare diseases for this patient.",
        "Format each result on a SINGLE LINE exactly like this:",
        "DISEASE: <name> | ORDO: ORPHA:<number> | MATCH: <phenotypes> | CONFIDENCE: <high/medium/low>",
        "",
        "Example:",
        "DISEASE: Friedreich Ataxia | ORDO: ORPHA:95 | MATCH: cerebellar ataxia, anemia | CONFIDENCE: high",
        "DISEASE: Wilson Disease | ORDO: ORPHA:905 | MATCH: developmental delay, anemia | CONFIDENCE: medium",
        "DISEASE: Gaucher Disease | ORDO: ORPHA:355 | MATCH: anemia, developmental delay | CONFIDENCE: low",
        "",
        f"Now list {top_k} different diseases:",
    ]

    content = "\n".join(content_parts)

    # Mistral/BioMistral instruction format
    return f"[INST] {content} [/INST]"


# ── Retrieval output parser ───────────────────────────────────────────────────


def find_disease_in_profiles(
    ordo_id: str,
    disease_name: str,
    disease_profiles: Dict[str, dict],
) -> Tuple[str, str, bool]:
    """
    Find a disease in profiles by ORPHA ID first, then fuzzy name match.

    Returns:
        (matched_id, label, validated) — validated=True if found in profiles.
    """
    if ordo_id in disease_profiles:
        return ordo_id, disease_profiles[ordo_id].get("label", disease_name), True

    normalized_id = ordo_id.replace(":", "")
    for pid in disease_profiles:
        if pid.replace(":", "") == normalized_id:
            return pid, disease_profiles[pid].get("label", disease_name), True

    disease_name_lower = disease_name.lower()
    for pid, profile in disease_profiles.items():
        label = profile.get("label", "").lower()
        if disease_name_lower in label or label in disease_name_lower:
            return pid, profile.get("label", disease_name), True

    return ordo_id, disease_name, False


def parse_retrieval_output(
    generated_text: str,
    disease_profiles: Dict[str, dict],
    model_name: str,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Parse LLM generated text into structured disease results.

    Handles common LLM formatting variations and validates each result
    against the disease profiles to flag hallucinated diseases.
    """
    if not generated_text:
        return []

    # Fix common LLM formatting issues
    generated_text = re.sub(r"ORPHA(\d+)", r"ORPHA:\1", generated_text)
    generated_text = re.sub(r"OMIM(\d+)", r"OMIM:\1", generated_text)
    generated_text = re.sub(r"\[SOLUTION\]", "", generated_text)
    generated_text = re.sub(
        r"\[INST\].*?\[/INST\]", "", generated_text, flags=re.DOTALL
    )

    normalized = " ".join(
        line.strip() for line in generated_text.splitlines() if line.strip()
    )

    pattern = re.compile(
        r"DISEASE:\s*(.+?)\s*\|\s*ORDO:\s*([\w:]+)\s*\|\s*MATCH:\s*(.+?)\s*\|\s*\w+:\s*([\w\s]+?)(?=DISEASE:|$)",
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

        # Skip duplicates
        if ordo_id in seen_ids:
            continue
        seen_ids.add(ordo_id)

        matched_id, label, validated = find_disease_in_profiles(
            ordo_id, disease_name, disease_profiles
        )

        results.append(
            {
                "rank": rank,
                "disease_name": label,
                "ordo_id": matched_id,
                "matched_phenotypes": matched_phenotypes,
                "confidence": confidence,
                "validated_against_profiles": validated,
                "model": model_name,
                "method": "llm_retrieval",
            }
        )

        rank += 1
        if rank > top_k:
            break

    return results


# ── Retrieval entry point ─────────────────────────────────────────────────────


def retrieve_diseases_llm(
    patient: dict,
    hpo_labels: Dict[str, str],
    disease_profiles: Dict[str, dict],
    model_name: str,
    top_k: int = TOP_K,
) -> Tuple[List[dict], object]:
    """
    Use an LLM to directly retrieve and rank rare diseases from patient HPO terms.

    Loads the model, generates results, then returns both the results and the
    pipeline object so the caller can unload it when ready.

    Args:
        patient:          Patient dict with hpo_terms.
        hpo_labels:       HPO ID → label mapping.
        disease_profiles: Known disease profiles for validation.
        model_name:       HuggingFace model identifier.
        top_k:            Number of diseases to return.

    Returns:
        Tuple of (results list, pipeline object).
        Caller should call unload_pipeline(pipe) when done.
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

    results = parse_retrieval_output(generated, disease_profiles, model_name, top_k)

    n_validated = sum(r["validated_against_profiles"] for r in results)
    print(
        f"[llm] Found {len(results)} diseases ({n_validated} validated against profiles)"
    )

    return results, pipe


# ── Explanation prompt builder ────────────────────────────────────────────────


def build_explanation_prompt(
    patient: dict,
    disease: dict,
    hpo_labels: Dict[str, str],
    transformer_score: float = None,
    transformer_rank: int = None,
) -> str:
    """Build a structured clinical reasoning prompt for one patient-disease pair."""
    patient_terms = patient.get("hpo_terms", [])
    disease_terms = disease.get("hpo_terms", [])

    patient_labels = [hpo_labels.get(t, t) for t in patient_terms]
    disease_labels = [hpo_labels.get(t, t) for t in disease_terms]

    disease_label = disease.get("label", "Unknown disease")
    disease_desc = (disease.get("merged_description") or "").strip()

    patient_set = set(patient_terms)
    disease_set = set(disease_terms)
    matching_ids = patient_set & disease_set
    matching_labels = [hpo_labels.get(t, t) for t in matching_ids]
    missing_from_disease = [hpo_labels.get(t, t) for t in patient_set - disease_set]

    prompt_parts = [
        "You are a rare disease clinical expert. Your task is to evaluate whether "
        "a patient's phenotype profile matches a candidate rare disease.",
        "",
        "---",
        f"PATIENT PHENOTYPES: {', '.join(patient_labels) if patient_labels else 'None listed'}",
        "---",
        f"CANDIDATE DISEASE: {disease_label}",
    ]

    if disease_desc:
        prompt_parts += [f"DISEASE DESCRIPTION: {disease_desc[:400]}"]

    if disease_labels:
        prompt_parts += [f"DISEASE PHENOTYPES: {', '.join(disease_labels[:20])}"]

    if transformer_score is not None:
        prompt_parts += [
            "",
            f"Embedding similarity score: {transformer_score:.4f} (rank #{transformer_rank})",
        ]

    prompt_parts += [
        "",
        "Pre-computed phenotype overlap for your reference:",
        f"  Directly matched: {', '.join(matching_labels) if matching_labels else 'None'}",
        f"  In patient but not in disease: {', '.join(missing_from_disease) if missing_from_disease else 'None'}",
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


def extract_explanation(generated_text: str) -> str:
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
    short = ". ".join(s.strip() for s in sentences[:3] if s.strip())
    return short + "." if short and not short.endswith(".") else short


# ── Explanation entry point ───────────────────────────────────────────────────


def explain_top_results(
    patient: dict,
    transformer_results: List[dict],
    disease_profiles: Dict[str, dict],
    hpo_labels: Dict[str, str],
    model_name: str = EXPLAINER_MODEL,
    top_k: int = TOP_K_RERANK,
) -> List[dict]:
    """
    Add structured LLM explanations to transformer top-K results.

    Loads the explainer model once, explains all top_k results,
    then unloads the model to free GPU memory.
    """
    print(f"\n[llm] Loading explainer: {model_name}")
    with timer("load explainer"):
        pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_EXPLAINER)

    candidates = transformer_results[:top_k]
    explained = []

    for i, result in enumerate(candidates):
        disease_id = (
            result.get("canonical_disease_id")
            or result.get("disease_id")
            or result.get("ordo_id")
        )

        if not disease_id or disease_id not in disease_profiles:
            result["llm_explanation"] = "Disease profile not found."
            explained.append(result)
            continue

        disease = disease_profiles[disease_id]
        label = disease.get("label", disease_id)

        print(f"  [llm] {i + 1}/{len(candidates)}: {label}")

        with timer(f"explain {label[:40]}"):
            prompt = build_explanation_prompt(
                patient=patient,
                disease=disease,
                hpo_labels=hpo_labels,
                transformer_score=result.get("score"),
                transformer_rank=result.get("rank"),
            )
            generated = query_hf(prompt, pipe, max_tokens=MAX_NEW_TOKENS_EXPLAINER)
            explanation = extract_explanation(generated)

        result["llm_explanation"] = explanation
        result["explainer_model"] = model_name
        explained.append(result)

    unload_pipeline(pipe)
    return explained
