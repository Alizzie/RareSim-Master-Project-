"""Patient loading with optional HPO extraction from raw text."""

from pathlib import Path
from raresim.hpo_extraction import build_patient_profile
from raresim.types.schemas import PatientProfile
from raresim.utils.io import load_json


def load_patient(path: Path) -> PatientProfile:
    """Load a patient profile from a JSON file and return it as a PatientProfile object."""
    data = load_json(path)
    return PatientProfile(
        patient_id=data["patient_id"],
        raw_text=data.get("raw_text", ""),
        hpo_terms=set(data.get("hpo_terms", [])),
        propagated_hpo_terms=set(data.get("propagated_hpo_terms", [])),
    )


def load_patient_with_extraction(
    patient_path: Path,
    hpo_labels: dict,
    methods: list = ("dictionary",),
) -> PatientProfile:
    """
    Load a patient profile from a JSON file, extracting HPO terms from raw text if necessary.

    If the patient JSON already has hpo_terms, loads directly.
    Otherwise runs build_patient_profile to extract HPO terms from raw_text, then constructs a PatientProfile.
    """

    data = load_json(patient_path)

    if data.get("hpo_terms"):
        return load_patient(patient_path)

    raw_text = data.get("raw_text", "").strip()
    if not raw_text:
        raise ValueError(
            f"Patient {data.get('patient_id')} has neither " "hpo_terms nor raw_text."
        )

    enriched, _ = build_patient_profile(
        patient_id=data["patient_id"],
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=list(methods),
    )
    # convert enriched dict to PatientProfile
    return PatientProfile(
        patient_id=enriched["patient_id"],
        raw_text=enriched["raw_text"],
        hpo_terms=set(enriched["hpo_terms"]),
        propagated_hpo_terms=set(enriched.get("propagated_hpo_terms", [])),
    )
