import re
import requests
from typing import Dict, List

from llm_config import (
    DO_SAMPLE,
    LLM_BACKEND,
    MAX_NEW_TOKENS_RETRIEVAL,
    OLLAMA_URL,
    RETRIEVAL_MODEL,
    TEMPERATURE,
    TOP_K,
)

"""
LLM-based disease retrieval.

Supports two backends:
- Ollama  : local server, no loading delay, recommended for development
- HF      : HuggingFace transformers, best biomedical quality on GPU server

Switch backend in llm_config.py:
    LLM_BACKEND = "ollama"   ← development
    LLM_BACKEND = "hf"       ← GPU server with BioMistral

How it works:
1. Convert patient HPO IDs → readable labels
2. Build a structured prompt with patient phenotypes
3. LLM generates a ranked list of candidate diseases
4. Parse generated text into structured results

Limitation:
- LLM may hallucinate disease names or ORDO IDs
- Output is validated against known disease profiles
- Use transformer pipeline for fast retrieval, this for reasoning
"""


# ── Backend loaders ───────────────────────────────────────────────────────────

def query_ollama(
    prompt: str,
    model: str,
    max_tokens: int = MAX_NEW_TOKENS_RETRIEVAL,
) -> str:
    """
    Query a local Ollama server.
    Make sure ollama is running: ollama serve
    Pull the model first: ollama pull mistral
    """
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
            "[llm_retriever] Ollama server not running.\n"
            "Start it with: ollama serve\n"
            "Pull model with: ollama pull mistral"
        )


def load_hf_pipeline(model_name: str):
    """
    Load a HuggingFace text-generation pipeline.
    Uses 4-bit quantization if bitsandbytes is available.
    Recommended for GPU server only.
    """
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
            print(f"[llm_retriever] Loaded {model_name} with 4-bit quantization")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                use_safetensors=False,
            )
            print(f"[llm_retriever] Loaded {model_name} without quantization")

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS_RETRIEVAL,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
        )
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Install with: pip install transformers torch"
        )


def query_llm(prompt: str, model_name: str, pipe=None) -> str:
    """Dispatch query to the correct backend based on LLM_BACKEND."""
    if LLM_BACKEND == "ollama":
        return query_ollama(prompt, model_name)

    if LLM_BACKEND == "hf":
        if pipe is None:
            pipe = load_hf_pipeline(model_name)
        output = pipe(prompt)[0]["generated_text"]
        return output[len(prompt):].strip()

    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'ollama' or 'hf'.")


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_retrieval_prompt(
    patient: dict,
    hpo_labels: Dict[str, str],
    top_k: int = TOP_K,
) -> str:
    """
    Build a structured prompt for disease retrieval.
    Converts HPO IDs to readable labels and asks the LLM
    to rank the most likely rare diseases.
    """
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


# ── Output parser ─────────────────────────────────────────────────────────────

def find_disease_in_profiles(
    ordo_id: str,
    disease_name: str,
    disease_profiles: Dict[str, dict],
) -> tuple:
    """
    Try to find a disease in profiles by ORPHA ID first,
    then fall back to fuzzy name matching.
    Returns (matched_id, label, validated)
    """
    # Stage 1: exact ID match
    if ordo_id in disease_profiles:
        return ordo_id, disease_profiles[ordo_id].get("label", disease_name), True

    # Stage 2: normalize and try ID variations e.g. ORPHA:208570 → ORPHA208570
    normalized_id = ordo_id.replace(":", "")
    for pid in disease_profiles:
        if pid.replace(":", "") == normalized_id:
            return pid, disease_profiles[pid].get("label", disease_name), True

    # Stage 3: fuzzy name match
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
    """
    Parse LLM generated text into structured disease results.
    Validates disease names against known disease profiles.
    """
    if not generated_text:
        return []

    # normalize ORPHA/OMIM ids — add colon if missing
    generated_text = re.sub(r"ORPHA(\d+)", r"ORPHA:\1", generated_text)
    generated_text = re.sub(r"OMIM(\d+)", r"OMIM:\1", generated_text)

    # join lines to handle multiline entries
    normalized = " ".join(
        line.strip() for line in generated_text.splitlines() if line.strip()
    )

    # pattern handles:
    # - any label before confidence (CONFIDENCE, CONFIDANCE, CONF etc.)
    # - multi-word confidence values (very high, very low)
    # - stops at next DISEASE: entry or end of string
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


# ── Main entry point ──────────────────────────────────────────────────────────

def retrieve_diseases_llm(
    patient: dict,
    hpo_labels: Dict[str, str],
    disease_profiles: Dict[str, dict],
    model_name: str = RETRIEVAL_MODEL,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Use an LLM to directly retrieve and rank rare diseases
    from patient HPO terms via text generation.

    Args:
        patient:          Patient dict with hpo_terms.
        hpo_labels:       HPO ID → label mapping.
        disease_profiles: Known disease profiles for validation.
        model_name:       Model name (Ollama or HF depending on backend).
        top_k:            Number of diseases to return.

    Returns:
        List of ranked disease dicts with explanations.
    """
    prompt = build_retrieval_prompt(patient, hpo_labels, top_k)

    print(f"[llm_retriever] Running disease retrieval "
          f"(backend={LLM_BACKEND}, model={model_name})...")

    generated = query_llm(prompt, model_name)

    print("\n--- RAW LLM OUTPUT ---")
    print(generated)
    print("--- END OUTPUT ---\n")

    results = parse_retrieval_output(generated, disease_profiles, top_k)

    print(f"[llm_retriever] Found {len(results)} diseases "
          f"({sum(r['validated_against_profiles'] for r in results)} validated)")

    return results
