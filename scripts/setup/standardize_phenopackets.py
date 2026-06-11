"""
This script standardizes phenopacket JSON files into a consistent format of [[HP terms], [disease codes]].
Make sure to upload the phenopacket files to the "phenopackets/" directory before running.
This version is using 0.1.26 folder from all_phenopackets.zip available at
https://github.com/monarch-initiative/phenopacket-store/releases

Usage: python evaluation/standardize_phenopackets.py
"""

import json
from pathlib import Path
from raresim.utils.paths import PHENOPACKETS_DIR, STANDARDIZED_PHENOPACKETS_DIR


def phenopacket_to_standard(phenopacket: dict) -> list:
    """Convert a single phenopacket to [[HP terms], [disease codes]]"""

    # Extract HPO terms (exclude negated ones)
    hpo_terms = []
    for feature in phenopacket.get("phenotypicFeatures", []):
        if not feature.get("excluded", False):
            hpo_id = feature["type"]["id"]
            if hpo_id.startswith("HP:"):
                hpo_terms.append(hpo_id)

    # Extract disease codes from interpretations
    disease_codes = []
    for interp in phenopacket.get("interpretations", []):
        disease_id = interp.get("diagnosis", {}).get("disease", {}).get("id")
        if disease_id:
            disease_codes.append(disease_id)

    # Also check diseases field as fallback
    if not disease_codes:
        for disease in phenopacket.get("diseases", []):
            disease_id = disease.get("term", {}).get("id")
            if disease_id:
                disease_codes.append(disease_id)

    # Sort and deduplicate
    hpo_terms = sorted(set(hpo_terms))
    disease_codes = sorted(set(disease_codes))

    return [hpo_terms, disease_codes]


def process_phenopackets(input_folder: Path):
    """
    input_path: path to a single .json file OR a directory of phenopackets
    output_path: path to save the standardized output
    """

    files = list(input_folder.rglob("*.json"))
    if not files:
        print(f"No JSON files found in {input_folder}")
        return

    print(f"Found {len(files)} phenopacket(s)")

    results = []
    skipped = 0

    for f in sorted(files):
        data = json.loads(f.read_text())
        cases = (
            data if isinstance(data, list) else [data]
        )  # Handle both single and list of phenopackets

        for case in cases:
            hpo_terms, disease_codes = phenopacket_to_standard(case)

            if not hpo_terms or not disease_codes:
                print(
                    f"  Skipping {case.get('id', '?')} — missing HPO terms or disease codes"
                )
                skipped += 1
                continue

            results.append([hpo_terms, disease_codes])

    # Save output
    output_path = STANDARDIZED_PHENOPACKETS_DIR / f"{input_folder.name}.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Done: {len(results)} cases saved, {skipped} skipped → {output_path}")


if __name__ == "__main__":

    subfolders = [f for f in PHENOPACKETS_DIR.iterdir() if f.is_dir()]

    if not subfolders:
        process_phenopackets(PHENOPACKETS_DIR)
    else:
        for folder in sorted(subfolders):
            print(f"Processing folder: {folder.name}")
            process_phenopackets(folder)
