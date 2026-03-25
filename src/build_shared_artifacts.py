import json
from dataclasses import asdict
from pathlib import Path

from config import (
    APPLY_TRUE_PATH_RULE,
    EXAMPLE_PATIENT,
    HPO_PATH,
    HPOA_PATH,
    HOOM_PATH,
    MONDO_PATH,
    ORDO_PATH,
    OUTPUT_DIR,
)
from disease_profiles import build_disease_profiles
from hpo_utils import compute_ancestors, propagate_hpo_terms
from ic import compute_information_content, compute_term_frequencies
from loaders import (
    load_hpo_owl,
    load_hpoa_annotations,
    load_hoom_metadata,
    load_mondo_metadata,
    load_ordo_metadata,
)
from mapping_utils import build_orpha_mapping_index
from normalizers import normalize_hpo_id, normalize_owl_local_id
from schemas import PatientProfile
'''Main script to build shared artifacts for the project, including disease profiles, HPO term frequencies, information content values, and an example patient profile.'''

def save_json(data, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_terms: list[str],
    hpo_labels: dict[str, str],
    hpo_ancestors: dict[str, set[str]],
) -> PatientProfile:
    normalized_terms = {
        term for term in (normalize_hpo_id(x) for x in hpo_terms)
        if term is not None and term in hpo_labels
    }

    propagated_terms = propagate_hpo_terms(normalized_terms, hpo_ancestors)

    return PatientProfile(
        patient_id=patient_id,
        raw_text=raw_text,
        hpo_terms=normalized_terms,
        propagated_hpo_terms=propagated_terms,
    )


def main() -> None:
    print("Loading HPO...")
    hpo_labels, hpo_parents = load_hpo_owl(HPO_PATH)

    print("Computing HPO ancestors...")
    hpo_ancestors = compute_ancestors(hpo_parents)

    print("Loading HPOA annotations...")
    hpoa_records = load_hpoa_annotations(HPOA_PATH)

    print("Loading ORDO metadata...")
    ordo_metadata = load_ordo_metadata(ORDO_PATH, normalize_owl_local_id)

    print("Loading MONDO metadata...")
    mondo_metadata = load_mondo_metadata(MONDO_PATH, normalize_owl_local_id)

    print("Loading HOOM metadata...")
    hoom_metadata = load_hoom_metadata(HOOM_PATH, normalize_owl_local_id)

    print("Building OMIM/MONDO -> ORPHA mapping index...")
    mapping_index = build_orpha_mapping_index(
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata=hoom_metadata,
    )
    print(f"Mapping index size: {len(mapping_index)}")

    print("Building disease profiles with ORPHA canonicalization...")
    disease_profiles = build_disease_profiles(
        hpoa_records=hpoa_records,
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata=hoom_metadata,
        mapping_index=mapping_index,
        apply_true_path_rule=APPLY_TRUE_PATH_RULE,
    )

    print("Computing term frequencies and IC...")
    term_frequencies = compute_term_frequencies(disease_profiles)
    ic_values = compute_information_content(
        term_frequencies=term_frequencies,
        total_diseases=len(disease_profiles),
    )

    print("Building example patient profile...")
    patient = build_patient_profile(
        patient_id=EXAMPLE_PATIENT["patient_id"],
        raw_text=EXAMPLE_PATIENT["raw_text"],
        hpo_terms=EXAMPLE_PATIENT["hpo_terms"],
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
    )

    print("Saving outputs...")
    disease_profiles_json = {}
    for disease_id, profile in disease_profiles.items():
        row = asdict(profile)
        row["hpo_terms"] = sorted(profile.hpo_terms)
        row["propagated_hpo_terms"] = sorted(profile.propagated_hpo_terms)
        disease_profiles_json[disease_id] = row

    patient_json = asdict(patient)
    patient_json["hpo_terms"] = sorted(patient.hpo_terms)
    patient_json["propagated_hpo_terms"] = sorted(patient.propagated_hpo_terms)

    save_json(disease_profiles_json, OUTPUT_DIR / "disease_profiles.json")
    save_json(hpo_labels, OUTPUT_DIR / "hpo_labels.json")
    save_json(
        {k: sorted(v) for k, v in hpo_ancestors.items()},
        OUTPUT_DIR / "hpo_ancestors.json",
    )
    save_json(term_frequencies, OUTPUT_DIR / "term_frequencies.json")
    save_json(ic_values, OUTPUT_DIR / "information_content.json")
    save_json(patient_json, OUTPUT_DIR / "example_patient.json")
    save_json(mapping_index, OUTPUT_DIR / "orpha_mapping_index.json")

    print("Done.")
    print(f"Disease profiles built: {len(disease_profiles)}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
