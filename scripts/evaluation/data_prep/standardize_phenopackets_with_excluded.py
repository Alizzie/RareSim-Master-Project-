"""
Standardize phenopacket JSON files into a RareSim format that preserves
both observed and excluded HPO terms.

Output format:

[
  {
    "hpo_terms": ["HP:..."],
    "excluded_hpo_terms": ["HP:..."],
    "disease_codes": ["ORPHA:...", "OMIM:..."]
  }
]

This dataset is intended for methods that explicitly use excluded
phenotypes, such as penalized Jaccard similarity. Existing positive-only
standardized datasets do not need to be changed.

Examples:

    python scripts/evaluation/data_prep/standardize_phenopackets_with_excluded.py \
        --input data/datasets/phenopackets/raw \
        --output data/datasets/phenopackets/standardized_to_json/0.1.27_with_excluded.json

    python scripts/evaluation/data_prep/standardize_phenopackets_with_excluded.py \
        --input data/datasets/GA4GH_phenopackets/raw \
        --output data/datasets/GA4GH_phenopackets/standardized_to_json/ga4gh_phenopackets_with_excluded.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from raresim.utils.paths import ORPHA_MAPPING_INDEX


StandardizedCase = dict[str, list[str]]


def load_orpha_mapping() -> dict[str, str]:
    """Load disease-code to ORPHA mapping if available."""
    if not ORPHA_MAPPING_INDEX.exists():
        print(f"Warning: ORPHA mapping not found: {ORPHA_MAPPING_INDEX}")
        return {}

    data = json.loads(ORPHA_MAPPING_INDEX.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected ORPHA mapping to contain a JSON object: "
            f"{ORPHA_MAPPING_INDEX}"
        )

    return {str(key): str(value) for key, value in data.items()}


def extract_hpo_terms(phenopacket: dict[str, Any]) -> list[str]:
    """Extract observed, non-excluded HPO terms."""
    hpo_terms: list[str] = []

    for feature in phenopacket.get("phenotypicFeatures", []):
        if not isinstance(feature, dict):
            continue

        if feature.get("excluded", False):
            continue

        feature_type = feature.get("type", {})
        if not isinstance(feature_type, dict):
            continue

        hpo_id = feature_type.get("id")

        if isinstance(hpo_id, str) and hpo_id.startswith("HP:"):
            hpo_terms.append(hpo_id)

    return sorted(set(hpo_terms))


def extract_excluded_hpo_terms(
    phenopacket: dict[str, Any],
) -> list[str]:
    """Extract explicitly excluded HPO terms."""
    excluded_hpo_terms: list[str] = []

    for feature in phenopacket.get("phenotypicFeatures", []):
        if not isinstance(feature, dict):
            continue

        if not feature.get("excluded", False):
            continue

        feature_type = feature.get("type", {})
        if not isinstance(feature_type, dict):
            continue

        hpo_id = feature_type.get("id")

        if isinstance(hpo_id, str) and hpo_id.startswith("HP:"):
            excluded_hpo_terms.append(hpo_id)

    return sorted(set(excluded_hpo_terms))


def extract_disease_codes(phenopacket: dict[str, Any]) -> list[str]:
    """Extract disease codes from interpretations or diseases fallback."""
    disease_codes: list[str] = []

    interpretations = phenopacket.get("interpretations", [])
    if isinstance(interpretations, list):
        for interpretation in interpretations:
            if not isinstance(interpretation, dict):
                continue

            diagnosis = interpretation.get("diagnosis", {})
            if not isinstance(diagnosis, dict):
                continue

            disease = diagnosis.get("disease", {})
            if not isinstance(disease, dict):
                continue

            disease_id = disease.get("id")

            if isinstance(disease_id, str) and disease_id:
                disease_codes.append(disease_id)

    if not disease_codes:
        diseases = phenopacket.get("diseases", [])
        if isinstance(diseases, list):
            for disease in diseases:
                if not isinstance(disease, dict):
                    continue

                term = disease.get("term", {})
                if not isinstance(term, dict):
                    continue

                disease_id = term.get("id")

                if isinstance(disease_id, str) and disease_id:
                    disease_codes.append(disease_id)

    return sorted(set(disease_codes))


def phenopacket_to_standard(
    phenopacket: dict[str, Any],
    orpha_mapping: dict[str, str],
) -> StandardizedCase:
    """Convert one phenopacket to the negative-aware RareSim format."""
    hpo_terms = extract_hpo_terms(phenopacket)
    excluded_hpo_terms = extract_excluded_hpo_terms(phenopacket)
    disease_codes = extract_disease_codes(phenopacket)

    mapped_orpha_codes = [
        orpha_mapping[code]
        for code in disease_codes
        if code in orpha_mapping
    ]

    all_disease_codes = sorted(set(disease_codes + mapped_orpha_codes))

    return {
        "hpo_terms": hpo_terms,
        "excluded_hpo_terms": excluded_hpo_terms,
        "disease_codes": all_disease_codes,
    }


def load_json_cases(file_path: Path) -> list[Any]:
    """Load one JSON file and normalize its root value to a list."""
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def standardize_phenopackets(input_path: Path, output_path: Path) -> None:
    """Standardize one phenopacket JSON file or a folder of JSON files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.json"))

    if not files:
        print(f"No JSON files found in {input_path}")
        return

    print(f"Found {len(files)} JSON file(s)")

    orpha_mapping = load_orpha_mapping()

    results: list[StandardizedCase] = []
    skipped = 0
    files_with_errors = 0

    for file_path in files:
        try:
            cases = load_json_cases(file_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable JSON file {file_path}: {error}")
            files_with_errors += 1
            continue

        for case in cases:
            if not isinstance(case, dict):
                print(
                    f"Skipping non-phenopacket object in {file_path}: "
                    f"{type(case).__name__}"
                )
                skipped += 1
                continue

            standardized_case = phenopacket_to_standard(
                case,
                orpha_mapping,
            )

            if (
                not standardized_case["hpo_terms"]
                or not standardized_case["disease_codes"]
            ):
                print(
                    f"Skipping {case.get('id', file_path.name)} "
                    f"— missing positive HPO terms or disease codes"
                )
                skipped += 1
                continue

            results.append(standardized_case)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(
        f"Done: {len(results)} cases saved, "
        f"{skipped} cases skipped, "
        f"{files_with_errors} files unreadable -> {output_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Standardize phenopackets while preserving both positive "
            "and excluded HPO terms."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Phenopacket JSON file or folder containing phenopacket "
            "JSON files."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output negative-aware standardized JSON file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    standardize_phenopackets(args.input, args.output)
