import json
import os
import glob
'''
This script standardizes phenopacket JSON files into a consistent format of [[HP terms], [disease codes]].
Make sure to upload the phenopacket files to the "phenopackets/" directory before running.
This version is using 0.1.26 folder from all_phenopackets.zip available at
https://github.com/monarch-initiative/phenopacket-store/releases

Usage: python evaluation/standardize_phenopackets.py \
    --input phenopackets/0.1.26/ \
    --output standardized.json
'''
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


def process_phenopackets(input_path: str, output_file: str):
    """
    input_path: path to a single .json file OR a directory of phenopackets
    output_file: path to save the standardized output
    """
    
    # Handle single file or directory
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "**/*.json"), recursive=True)
        files += glob.glob(os.path.join(input_path, "*.json"))
        files = list(set(files))  # deduplicate
    else:
        raise ValueError(f"Input path not found: {input_path}")
    
    print(f"Found {len(files)} phenopacket(s)")
    
    results = []
    skipped = 0
    
    for f in sorted(files):
        with open(f, "r") as fh:
            data = json.load(fh)
        
        # Handle both single phenopacket and list of phenopackets
        if isinstance(data, list):
            cases = data
        else:
            cases = [data]
        
        for case in cases:
            standardized = phenopacket_to_standard(case)
            hpo_terms, disease_codes = standardized
            
            if not hpo_terms or not disease_codes:
                print(f"  Skipping {case.get('id', '?')} — missing HPO terms or disease codes")
                skipped += 1
                continue
            
            results.append(standardized)
    
    # Save output
    with open(output_file, "w") as fh:
        json.dump(results, fh, indent=2)
    
    print(f"Done: {len(results)} cases saved, {skipped} skipped → {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to phenopacket JSON file or directory")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    process_phenopackets(args.input, args.output)
