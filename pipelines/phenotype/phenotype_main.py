import json
from pathlib import Path
from typing import Any

from phenotype_config import (
    EXTRACTION_METHODS,
    HPO_LABELS_PATH,
    OUTPUT_EXTRACTION_PATH,
    OUTPUT_PATIENT_PATH,
)
from phenotype_extractor import build_patient_profile

"""
Entrypoint for the phenotype extraction pipeline.

Reads raw patient text, runs extraction, and saves:
- example_patient_extracted.json   → patient profile (HPO term list)
- example_patient_hpo_extraction.json → full extraction provenance

"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    hpo_labels = load_json(HPO_LABELS_PATH)

    # Replace with real patient text or load from file
    raw_text = (
        "Patient with developmental delay, cerebellar ataxia, and anemia. "
        "No seizures reported."
    )

    # Choose methods: "dictionary", "scispacy", "biomedical_ner"
    methods = ["dictionary", "scispacy", "biomedical_ner"]

    print(f"Running phenotype extraction with methods: {methods}\n")

    patient, extracted_terms = build_patient_profile(
        patient_id="patient_001",
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
    )

    save_json(patient, OUTPUT_PATIENT_PATH)
    save_json(extracted_terms, OUTPUT_EXTRACTION_PATH)

    print(f"Extracted {len(extracted_terms)} HPO term(s):\n")
    for row in extracted_terms:
        print(
            f"  {row['hpo_id']} | {row['label']:<40} | "
            f"conf={row['confidence']:.2f} | "
            f"method={row['method']}"
        )

    print(f"\nPatient profile  → {OUTPUT_PATIENT_PATH}")
    print(f"Extraction detail → {OUTPUT_EXTRACTION_PATH}")


if __name__ == "__main__":
    main()
