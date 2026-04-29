import requests
from typing import Dict, List

from llm_config import (
    DO_SAMPLE,
    EXPLAINER_MODEL,
    LLM_BACKEND,
    MAX_NEW_TOKENS_EXPLAINER,
    OLLAMA_URL,
    TEMPERATURE,
    TOP_K_RERANK,
)

"""
LLM-based explanation and reasoning for disease-patient matches.

Supports two backends:
- Ollama  : local server, no loading delay, recommended for development
- HF      : HuggingFace transformers, better biomedical quality on GPU server

Switch backend in llm_config.py:
    LLM_BACKEND = "ollama"   ← development
    LLM_BACKEND = "hf"       ← GPU server with BioMistral

This module does NOT replace the transformer retrieval —
it runs on top of it to add interpretability/reasoning for our results.
"""


# ── Backend loaders ───────────────────────────────────────────────────────────

def query_ollama(prompt: str, model: str) -> str:
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
                    "num_predict": MAX_NEW_TOKENS_EXPLAINER,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "[llm_explainer] Ollama server not running.\n"
            "Start it with: ollama serve\n"
            "Pull model with: ollama pull mistral"
        )


def load_hf_pipeline(model_name: str):
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
            print(f"[llm_explainer] Loaded {model_name} with 4-bit quantization")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                use_safetensors=False,
            )
            print(f"[llm_explainer] Loaded {model_name} without quantization")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_EXPLAINER,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
        )
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Install with: pip install transformers torch"
        )


def query_llm(prompt: str, model_name: str, pipe=None) -> str:
    """Dispatch query to the correct backend."""
    if LLM_BACKEND == "ollama":
        return query_ollama(prompt, model_name)

    if LLM_BACKEND == "hf":
        if pipe is None:
            pipe = load_hf_pipeline(model_name)
        output = pipe(prompt)[0]["generated_text"]
        # guard against None
        if not output:
            return ""
        return output[len(prompt):].strip()

    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'ollama' or 'hf'.")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_explanation_prompt(
    patient: dict,
    disease: dict,
    hpo_labels: Dict[str, str],
    transformer_score: float = None,
    transformer_rank: int = None,
) -> str:
    """
    Build a structured reasoning prompt for one patient-disease pair.
    Optionally includes transformer similarity score for richer reasoning.
    """
    patient_labels = [
        hpo_labels.get(t, t) for t in patient.get("hpo_terms", [])
    ]
    disease_labels = [
        hpo_labels.get(t, t) for t in disease.get("hpo_terms", [])
    ]
    disease_label = disease.get("label", "Unknown disease")
    disease_desc = (disease.get("merged_description") or "").strip()

    prompt_parts = [
        "You are a rare disease expert specializing in clinical phenotyping.",
        "",
        f"Patient phenotypes: {', '.join(patient_labels)}",
        "",
        f"Candidate disease: {disease_label}",
    ]

    if disease_desc:
        prompt_parts += [f"Disease description: {disease_desc[:300]}"]

    if disease_labels:
        prompt_parts += [f"Disease phenotypes: {', '.join(disease_labels[:15])}"]

    if transformer_score is not None:
        prompt_parts += [
            "",
            f"Embedding similarity score: {transformer_score:.4f} "
            f"(rank #{transformer_rank})",
        ]

    prompt_parts += [
        "",
        "In 2-3 sentences, explain:",
        "1. Which patient phenotypes match this disease",
        "2. Which key disease phenotypes are absent in the patient",
        "3. Overall assessment: likely match / possible match / unlikely match",
        "",
        "Explanation:",
    ]

    return "\n".join(prompt_parts)


# ── Explanation extractor ─────────────────────────────────────────────────────

def extract_explanation(generated_text: str) -> str:
    """Extract and trim explanation to 3 sentences."""
    if not generated_text:
        return "No explanation generated."

    generated_text = generated_text.strip()
    sentences = generated_text.split(".")
    short = ". ".join(s.strip() for s in sentences[:3] if s.strip())
    return short + "." if short and not short.endswith(".") else short


# ── Main entry points ─────────────────────────────────────────────────────────

def explain_match(
    patient: dict,
    disease: dict,
    hpo_labels: Dict[str, str],
    model_name: str = EXPLAINER_MODEL,
    pipe=None,
    transformer_score: float = None,
    transformer_rank: int = None,
) -> str:
    """
    Generate a clinical explanation for one patient-disease match.

    Args:
        patient:           Patient dict with hpo_terms.
        disease:           Disease profile dict.
        hpo_labels:        HPO ID → label mapping.
        model_name:        Model name for the active backend.
        pipe:              Pre-loaded HF pipeline (HF backend only).
        transformer_score: Optional cosine similarity score from transformer.
        transformer_rank:  Optional rank from transformer pipeline.

    Returns:
        Explanation string (2-3 sentences).
    """
    prompt = build_explanation_prompt(
        patient, disease, hpo_labels, transformer_score, transformer_rank
    )
    generated = query_llm(prompt, model_name, pipe)
    return extract_explanation(generated)


def explain_top_results(
    patient: dict,
    transformer_results: List[dict],
    disease_profiles: Dict[str, dict],
    hpo_labels: Dict[str, str],
    model_name: str = EXPLAINER_MODEL,
    top_k: int = TOP_K_RERANK,
) -> List[dict]:
    """
    Add LLM explanations to transformer top-K results.

    Args:
        patient:              Patient dict with hpo_terms.
        transformer_results:  Top-K results from transformer pipeline.
        disease_profiles:     Full disease profile dict.
        hpo_labels:           HPO ID → label mapping.
        model_name:           Model name for the active backend.
        top_k:                How many results to explain.

    Returns:
        Same result list with added llm_explanation field.
    """
    # load HF pipeline once if using HF backend
    pipe = load_hf_pipeline(model_name) if LLM_BACKEND == "hf" else None

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
            f"[llm_explainer] {i + 1}/{len(candidates)}: "
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
