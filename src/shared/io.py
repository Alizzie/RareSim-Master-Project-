"""Shared I/O utilities for loading and saving data in the RareSim project."""

import json
from pathlib import Path
from core.schemas import PatientProfile
from shared.result import SimilarityResult


def load_json(input_path: Path) -> dict:
    """Load a JSON file from a given path and return the data as a dictionary."""
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, output_path: Path) -> None:
    """Save a dictionary or list as a JSON file to a given path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_patient(path: Path) -> PatientProfile:
    """Load a patient profile from a JSON file and return it as a PatientProfile object."""
    data = load_json(path)
    return PatientProfile(
        patient_id=data["patient_id"],
        raw_text=data.get("raw_text", ""),
        hpo_terms=set(data.get("hpo_terms", [])),
        propagated_hpo_terms=set(data.get("propagated_hpo_terms", [])),
    )


def save_results(results: dict[str, list[SimilarityResult]], path: Path) -> None:
    """Save similarity results to a JSON file, organized by method name as a whole."""
    save_json(
        {method: [r.to_dict() for r in rows] for method, rows in results.items()},
        path,
    )


def save_individual_results(results: list[SimilarityResult], path: Path) -> None:
    """Save for each method separately, with method name in the filename."""
    save_json(
        [r.to_dict() for r in results],
        path,
    )
