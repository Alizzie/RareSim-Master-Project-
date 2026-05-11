"""
LLM methods — disease retrieval and explanation.

Combines:
- llm_retriever : query LLM backends, build retrieval prompts, parse output
- llm_explainer : build explanation prompts, extract structured reasoning

Supports two backends:
- Ollama : local server, recommended for development - right now we use this for both retrieval and explanation since it's more lightweight and easier to set up
- HF     : HuggingFace transformers, best biomedical quality on GPU server

Switch backend in config.py: LLM_BACKEND = "ollama" | "hf"
"""

import re
import requests
from typing import Dict, List

from similarity_methods.llm.config import (
    DO_SAMPLE,
    EXPLAINER_MODEL,
    LLM_BACKEND,
    MAX_NEW_TOKENS_EXPLAINER,
    MAX_NEW_TOKENS_RETRIEVAL,
    OLLAMA_URL,
    RETRIEVAL_MODEL,
    TEMPERATURE,
    TOP_K,
    TOP_K_RERANK,
)


# ── Backend loaders ───────────────────────────────────────────────────────────


def query_ollama(
    prompt: str,
    model: str,
    max_tokens: int = MAX_NEW_TOKENS_RETRIEVAL,
) -> str:
    """Query a local Ollama server."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "[llm] Ollama server not running.\n"
            "Start it with: ollama serve\n"
            "Pull model with: ollama pull mistral"
        )


def load_hf_pipeline(model_name: str, max_new_tokens: int = MAX_NEW_TOKENS_RETRIEVAL):
    """Load a HuggingFace text-generation pipeline. GPU server only."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
            print(f"[llm] Loaded {model_name} with 4-bit quantization")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                use_safetensors=False,
            )
            print(f"[llm] Loaded {model_name} without quantization")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
        )
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Install with: pip install transformers torch"
        )


def query_llm(
    prompt: str,
    model_name: str,
    pipe=None,
    max_tokens: int = MAX_NEW_TOKENS_RETRIEVAL,
) -> str:
    """Dispatch query to the correct backend based on LLM_BACKEND."""
    if LLM_BACKEND == "ollama":
        return query_ollama(prompt, model_name, max_tokens)

    if LLM_BACKEND == "hf":
        if pipe is None:
            pipe = load_hf_pipeline(model_name, max_tokens)
        output = pipe(prompt)[0]["generated_text"]
        return output[len(prompt):].strip() if output else ""

    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'ollama' or 'hf'.")


# ── Retrieval prompt builder ──────────────────────────────────────────────────


def build_retrieval_prompt(
    patient: dict,
    hpo_labels: Dict[str, str],
    top_k: int = TOP_K,
) -> str:
    """Build a structured prompt for disease retrieval."""
    hpo_term_labels = [
        hpo_labels.get(t, t) for t in patient.get("hpo_terms", [])
    ]
    raw_text = patient.get("raw_text", "").strip()

    prompt_parts = [
        "You are a rare disease expert specializing in clinical phenotyping.",
        "Answer directly without disclaimers or preamble.",
        "",
    ]

    if raw_text:
        prompt_parts += [f"Clinical description: {raw_text}", ""]

    prompt_parts += [
        f"Patient phenotypes (HPO terms): {', '.join(hpo_term_labels)}",
        "",
        f"List the top {top_k} most likely rare diseases for this patient.",
        "Format each result on a SINGLE LINE exactly like this:",
        "DISEASE: <name> | ORDO: ORPHA:<number> | MATCH: <phenotypes> | CONFIDENCE: <level>",
        "",
        "Example:",
        "DISEASE: Friedreich Ataxia | ORDO: ORPHA:95 | MATCH: cerebellar ataxia, anemia | CONFIDENCE: high",
        "",
        "Results:",
    ]

    return "\n".join(prompt_parts)


# ── Retrieval output parser ───────────────────────────────────────────────────


def find_disease_in_profiles(
    ordo_id: str,
    disease_name: str,
    disease_profiles: Dict[str, dict],
) -> tuple:
    """
    Find a disease in profiles by ORPHA ID first, then fuzzy name match.
    Returns (matched_id, label, validated).
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
    top_k: int = TOP_K,
) -> List[dict]:
    """Parse LLM generated text into structured disease results."""
    if not generated_text:
        return []

    generated_text = re.sub(r"ORPHA(\d+)", r"ORPHA:\1", generated_text)
    generated_text = re.sub(r"OMIM(\d+)", r"OMIM:\1", generated_text)

    normalized = " ".join(
        line.strip() for line in generated_text.splitlines() if line.strip()
    )

    pattern = re.compile(
        r"DISEASE:\s*(.+?)\s*\|\s*ORDO:\s*([\w:]+)\s*\|\s*MATCH:\s*(.+?)\s*\|\s*\w+:\s*([\w\s]+?)(?=DISEASE:|$)",
        re.IGNORECASE,
    )

    results = []
    rank = 1

    for match in pattern.finditer(normalized):
        disease_name = match.group(1).strip()
        ordo_id = match.group(2).strip()
        matched_phenotypes = match.group(3).strip()
        confidence = match.group(4).strip().lower()

        matched_id, label, validated = find_disease_in_profiles(
            ordo_id, disease_name, disease_profiles
        )

        results.append({
            "rank": rank,
            "disease_name": label,
            "ordo_id": matched_id,
            "matched_phenotypes": matched_phenotypes,
            "confidence": confidence,
            "validated_against_profiles": validated,
            "model": RETRIEVAL_MODEL,
            "backend": LLM_BACKEND,
            "method": "llm_retrieval",
        })

        rank += 1
        if rank > top_k:
            break

    return results


# ── Retrieval entry point ─────────────────────────────────────────────────────


def retrieve_diseases_llm(
    patient: dict,
    hpo_labels: Dict[str, str],
    disease_profiles: Dict[str, dict],
    model_name: str = RETRIEVAL_MODEL,
    top_k: int = TOP_K,
) -> List[dict]:
    """Use an LLM to directly retrieve and rank rare diseases from patient HPO terms."""
    prompt = build_retrieval_prompt(patient, hpo_labels, top_k)

    print(f"[llm] Running disease retrieval "
          f"(backend={LLM_BACKEND}, model={model_name})...")

    generated = query_llm(prompt, model_name, max_tokens=MAX_NEW_TOKENS_RETRIEVAL)

    print("\n--- RAW LLM OUTPUT ---")
    print(generated)
    print("--- END OUTPUT ---\n")

    results = parse_retrieval_output(generated, disease_profiles, top_k)

    print(f"[llm] Found {len(results)} diseases "
          f"({sum(r['validated_against_profiles'] for r in results)} validated)")

    return results


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
        "which are missing, and any clinically important discrepancies. "
        "Be specific — use the actual phenotype names.]",
        "",
        "VERDICT: [STRONG MATCH / POSSIBLE MATCH / WEAK MATCH / UNLIKELY MATCH]",
        "",
        "VERDICT REASON: [One sentence justifying the verdict based on phenotype "
        "overlap quality and any key contradicting features.]",
        "",
        "CLINICAL REASONING:",
    ]

    return "\n".join(prompt_parts)


# ── Explanation extractor ─────────────────────────────────────────────────────


def extract_explanation(generated_text: str) -> str:
    """
    Extract and structure the explanation from LLM output.
    Falls back to first 3 sentences if structure is missing.
    """
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


# ── Explanation entry points ──────────────────────────────────────────────────


def explain_match(
    patient: dict,
    disease: dict,
    hpo_labels: Dict[str, str],
    model_name: str = EXPLAINER_MODEL,
    pipe=None,
    transformer_score: float = None,
    transformer_rank: int = None,
) -> str:
    """Generate a structured clinical explanation for one patient-disease match."""
    prompt = build_explanation_prompt(
        patient, disease, hpo_labels, transformer_score, transformer_rank
    )
    generated = query_llm(prompt, model_name, pipe, max_tokens=MAX_NEW_TOKENS_EXPLAINER)
    return extract_explanation(generated)


def explain_top_results(
    patient: dict,
    transformer_results: List[dict],
    disease_profiles: Dict[str, dict],
    hpo_labels: Dict[str, str],
    model_name: str = EXPLAINER_MODEL,
    top_k: int = TOP_K_RERANK,
) -> List[dict]:
    """Add structured LLM explanations to transformer top-K results."""
    pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_EXPLAINER) if LLM_BACKEND == "hf" else None

    candidates = transformer_results[:top_k]
    explained = []

    for i, result in enumerate(candidates):
        disease_id = result.get("canonical_disease_id") or result.get("disease_id")

        if not disease_id or disease_id not in disease_profiles:
            result["llm_explanation"] = "Disease profile not found."
            explained.append(result)
            continue

        disease = disease_profiles[disease_id]

        print(
            f"[llm] {i + 1}/{len(candidates)}: "
            f"{disease.get('label', disease_id)}"
        )

        explanation = explain_match(
            patient=patient,
            disease=disease,
            hpo_labels=hpo_labels,
            model_name=model_name,
            pipe=pipe,
            transformer_score=result.get("score"),
            transformer_rank=result.get("rank"),
        )

        result["llm_explanation"] = explanation
        result["explainer_model"] = model_name
        result["explainer_backend"] = LLM_BACKEND
        explained.append(result)

    return explained
