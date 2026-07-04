"""
Standardize phenopacket JSON files into RareSim format:

[
  [[HP terms], [disease codes]],
  [[HP terms], [disease codes]]
]

Works for any folder containing GA4GH/Monarch phenopacket JSON files.

Examples:

    python scripts/evaluation/data_prep/standardize_phenopackets.py \
        --input data/datasets/phenopackets/raw \
        --output data/datasets/phenopackets/standardized_to_json/0.1.27.json

    python scripts/evaluation/data_prep/standardize_phenopackets.py \
        --input data/datasets/GA4GH_phenopackets/raw \
        --output data/datasets/GA4GH_phenopackets/standardized_to_json/ga4gh_phenopackets.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from raresim.utils.paths import ORPHA_MAPPING_INDEX


def load_orpha_mapping() -> dict[str, str]:
    """Load disease-code to ORPHA mapping if available."""
    if not ORPHA_MAPPING_INDEX.exists():
        print(f"Warning: ORPHA mapping not found: {ORPHA_MAPPING_INDEX}")
        return {}

    data = json.loads(ORPHA_MAPPING_INDEX.read_text(encoding="utf-8"))

    return {str(key): str(value) for key, value in data.items()}


def extract_hpo_terms(phenopacket: dict[str, Any]) -> list[str]:
    """Extract non-excluded HPO terms from phenotypicFeatures."""
    hpo_terms = []

    for feature in phenopacket.get("phenotypicFeatures", []):
        if feature.get("excluded", False):
            continue

        hpo_id = feature.get("type", {}).get("id")

        if isinstance(hpo_id, str) and hpo_id.startswith("HP:"):
            hpo_terms.append(hpo_id)

    return sorted(set(hpo_terms))


def extract_disease_codes(phenopacket: dict[str, Any]) -> list[str]:
    """Extract disease codes from interpretations or diseases fallback."""
    disease_codes = []

    for interpretation in phenopacket.get("interpretations", []):
        disease_id = (
            interpretation
            .get("diagnosis", {})
            .get("disease", {})
            .get("id")
        )

        if isinstance(disease_id, str) and disease_id:
            disease_codes.append(disease_id)

    if not disease_codes:
        for disease in phenopacket.get("diseases", []):
            disease_id = disease.get("term", {}).get("id")

            if isinstance(disease_id, str) and disease_id:
                disease_codes.append(disease_id)

    return sorted(set(disease_codes))


def phenopacket_to_standard(
    phenopacket: dict[str, Any],
    orpha_mapping: dict[str, str],
) -> list[list[str]]:
    """Convert one phenopacket to [[HP terms], [disease codes]]."""
    hpo_terms = extract_hpo_terms(phenopacket)
    disease_codes = extract_disease_codes(phenopacket)

    mapped_orpha_codes = [
        orpha_mapping[code]
        for code in disease_codes
        if code in orpha_mapping
    ]

    disease_codes = sorted(set(disease_codes + mapped_orpha_codes))

    return [hpo_terms, disease_codes]


def standardize_phenopackets(input_path: Path, output_path: Path) -> None:
    """Standardize one phenopacket JSON file or a folder of JSON files."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.json"))

    if not files:
        print(f"No JSON files found in {input_path}")
        return

    print(f"Found {len(files)} JSON file(s)")

    orpha_mapping = load_orpha_mapping()

    results = []
    skipped = 0

    for file_path in files:
        data = json.loads(file_path.read_text(encoding="utf-8"))

        cases = data if isinstance(data, list) else [data]

        for case in cases:
            if not isinstance(case, dict):
                print(f"Skipping non-phenopacket object in {file_path}: {type(case)}")
                skipped += 1
                continue

            hpo_terms, disease_codes = phenopacket_to_standard(
                case,
                orpha_mapping,
            )

            if not hpo_terms or not disease_codes:
                print(
                    f"Skipping {case.get('id', file_path.name)} "
                    f"— missing HPO terms or disease codes"
                )
                skipped += 1
                continue

            results.append([hpo_terms, disease_codes])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Done: {len(results)} cases saved, {skipped} skipped -> {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Phenopacket JSON file or folder containing phenopacket JSON files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file in [[HP terms], [disease codes]] format.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    standardize_phenopackets(args.input, args.output)
