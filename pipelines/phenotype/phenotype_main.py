import json
from pathlib import Path
from typing import Any, Optional

from phenotype_config import (
    HPO_LABELS_PATH,
    HPO_SYNONYMS_PATH,
    OUTPUT_EXTRACTION_PATH,
    OUTPUT_PATIENT_PATH,
)
from phenotype_extractor import build_patient_profile

"""
Entrypoint for the phenotype extraction pipeline.

Available methods:
- "dictionary"     : fast baseline, exact HPO label matching
- "synonyms"       : dictionary + HPO synonym expansion (requires hpo_synonyms.json)
- "biomedical_ner" : d4data transformer NER + HPO label lookup

Active methods: dictionary + synonyms + biomedical_ner
Saves:
- example_patient_extracted.json      → patient profile (HPO term list)
- example_patient_hpo_extraction.json → full extraction provenance
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_synonyms_if_available(path: Path) -> Optional[dict]:
    """Load HPO synonyms if available, otherwise return None."""
    if path.exists():
        return load_json(path)
    print(
        f"[phenotype_main] hpo_synonyms.json not found at {path}\n"
        "Synonym method will be skipped. Generate it by re-running build_shared_artifacts.py."
    )
    return None


def main() -> None:
    hpo_labels = load_json(HPO_LABELS_PATH)
    hpo_synonyms = load_synonyms_if_available(HPO_SYNONYMS_PATH)

    # Replace with real patient text or load from file
    raw_text = (
        "Patient with developmental delay, cerebellar ataxia, and anemia. "
        "No seizures reported."
    )

    # Active methods
    methods = ["dictionary", "synonyms", "biomedical_ner"]

    print(f"Running phenotype extraction with methods: {methods}\n")

    patient, extracted_terms = build_patient_profile(
        patient_id="patient_001",
        raw_text=raw_text,
        hpo_labels=hpo_labels,
        methods=methods,
        hpo_synonyms=hpo_synonyms,
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

    print(f"\nPatient profile   → {OUTPUT_PATIENT_PATH}")
    print(f"Extraction detail → {OUTPUT_EXTRACTION_PATH}")


if __name__ == "__main__":
    main()
