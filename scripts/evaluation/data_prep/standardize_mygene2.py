"""
Standardize MyGene2 JSONL records into RareSim evaluation format:

[
  [[HP terms], [disease codes]],
  [[HP terms], [disease codes]]
]

Input file:
    data/datasets/shepherd/raw/mygene2_5.7.22.txt

Output file:
    data/datasets/shepherd/standardized/mygene2_5.7.22.json

The MyGene2 file is JSONL: one JSON object per line.
It is not a phenopacket file.

Usage:
    python scripts/evaluation/data_prep/standardize_mygene2.py
"""

import json
from pathlib import Path
from typing import Any
from raresim.utils.paths import SHEPHERD_DIR, STANDARDIZED_SHEPHERD_DIR


INPUT_PATH = SHEPHERD_DIR / "mygene2_5.7.22.txt"

OUTPUT_PATH = STANDARDIZED_SHEPHERD_DIR / "mygene2_5.7.22.json"


def unique_sorted(values: list[str]) -> list[str]:
    """Deduplicate and sort string values."""
    return sorted(set(values))


def normalize_orpha_ids(value: Any) -> list[str]:
    """Convert MyGene2 orpha_id values into ORPHA-prefixed disease codes."""
    if value is None or value == "":
        return []

    if isinstance(value, list):
        return [f"ORPHA:{x}" for x in value if x is not None and x != ""]

    return [f"ORPHA:{value}"]


def normalize_omim_id(value: Any) -> list[str]:
    """Convert MyGene2 omim value into an OMIM-prefixed disease code."""
    if value is None or value == "":
        return []

    return [f"OMIM:{value}"]


def mygene2_to_standard(case: dict[str, Any]) -> list[list[str]]:
    """Convert one MyGene2 case to [[HP terms], [disease codes]]."""
    hpo_terms = unique_sorted(
        [
            term
            for term in case.get("positive_phenotypes", [])
            if isinstance(term, str) and term.startswith("HP:")
        ]
    )

    disease_codes = unique_sorted(
        normalize_orpha_ids(case.get("orpha_id"))
        + normalize_omim_id(case.get("omim"))
    )

    return [hpo_terms, disease_codes]


def standardize_mygene2(input_path: Path, output_path: Path) -> None:
    """Read MyGene2 JSONL and write standardized RareSim input JSON."""
    results: list[list[list[str]]] = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            case = json.loads(line)
            hpo_terms, disease_codes = mygene2_to_standard(case)

            if not hpo_terms or not disease_codes:
                case_id = case.get("id", f"line_{line_number}")
                print(f"Skipping {case_id} — missing HPO terms or disease codes")
                skipped += 1
                continue

            results.append([hpo_terms, disease_codes])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Done: {len(results)} cases saved, {skipped} skipped -> {output_path}")


if __name__ == "__main__":
    standardize_mygene2(INPUT_PATH, OUTPUT_PATH)
