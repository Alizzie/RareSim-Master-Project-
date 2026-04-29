import json
from pathlib import Path
from typing import Any

from llm_config import (
    DISEASE_PROFILES_PATH,
    HPO_LABELS_PATH,
    LLM_BACKEND,
    LLM_DIR,
    PATIENT_PATH,
    RETRIEVAL_MODEL,
    TOP_K,
)
from llm_retriever import retrieve_diseases_llm

"""
Main entrypoint for the LLM-based disease retrieval pipeline.

Runs BioMistral (HF backend) or Mistral via Ollama (ollama backend)
to directly retrieve rare diseases from patient HPO terms.

Backend is controlled in llm_config.py:
    LLM_BACKEND = "ollama"  ← local development (recommended)
    LLM_BACKEND = "hf"      ← if we have a GPU server with BioMistral

Note: LLM explanation of transformer results is handled inside
transformer_pipeline.py when RUN_LLM_EXPLAINER = True.

Load patient:
    Swap load_json(PATIENT_PATH) → load_patient(PATIENT_PATH, hpo_labels)
    after phenotype_extractor branch merges to main.
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)

    # swap to load_patient() after phenotype branch merges to main
    patient = load_json(PATIENT_PATH)

    print(f"Patient:   {patient.get('patient_id')}")
    print(f"HPO terms: {patient.get('hpo_terms')}")
    print(f"Backend:   {LLM_BACKEND}")
    print(f"Model:     {RETRIEVAL_MODEL}\n")

    print("=" * 60)
    print("LLM Disease Retrieval")
    print("=" * 60)

    llm_results = retrieve_diseases_llm(
        patient=patient,
        hpo_labels=hpo_labels,
        disease_profiles=disease_profiles,
        model_name=RETRIEVAL_MODEL,
        top_k=TOP_K,
    )

    out_path = LLM_DIR / f"llm_retrieval_top{TOP_K}.json"
    save_json(llm_results, out_path)

    print(f"\nResults (top {TOP_K}):\n")
    for r in llm_results:
        validated = "✓" if r["validated_against_profiles"] else "?"
        print(
            f"  rank={r['rank']:>2} | {validated} | "
            f"{r['ordo_id']:<15} | "
            f"conf={r['confidence']:<8} | "
            f"{r['disease_name']}"
        )

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
    