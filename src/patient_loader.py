from pathlib import Path
from typing import Optional
import json

"""
Shared patient loader for all pipelines.

Handles both cases:
- Patient already has HPO terms → use directly
- Patient has only raw text → run phenotype extractor first

MAKE SURE TO CHANGE patient = load_json(PATIENT_PATH) TO patient = load_patient(PATIENT_PATH, hpo_labels) IN ALL PIPELINES with from src.patient_loader import load_patient.
"""

def load_patient(
    patient_path: Path,
    hpo_labels: dict,
    methods: list = ("dictionary",),
) -> dict:
    """
    Load a patient JSON and ensure it has HPO terms.

    If hpo_terms already exist → return as-is.
    If only raw_text exists   → run phenotype extraction first.
    """
    with patient_path.open("r", encoding="utf-8") as f:
        patient = json.load(f)

    # Case 1: HPO terms already present — skip extractor
    if patient.get("hpo_terms"):
        return patient

    # Case 2: only raw text — run extractor
    raw_text = patient.get("raw_text", "").strip()
    if not raw_text:
        raise ValueError(
            f"Patient {patient.get('patient_id')} has neither "
            "hpo_terms nor raw_text."
        )

    from pipelines.phenotype.phenotype_extractor import build_patient_profile

    enriched_patient, _ = build_patient_profile(
        patient_id=patient["patient_id"],
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
    )
    return enriched_patient


# def load_patient_from_input method can be added later for GUI/API input.