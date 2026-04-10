import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from config import (
    APPLY_TRUE_PATH_RULE,
    EXAMPLE_PATIENT,
    HPO_PATH,
    HPOA_PATH,
    HOOM_PATH,
    MONARCH_DISEASE_TO_HPO_PATH,
    MONDO_PATH,
    ORDO_PATH,
    ORPHADATA_PRODUCT4_PATH,
    OUTPUT_DIR,
)
from disease_profiles import (
    build_canonical_disease_profiles,
    expand_alias_profiles,
)
from hpo_utils import compute_ancestors, propagate_hpo_terms
from ic import compute_information_content, compute_term_frequencies
from loaders import (
    load_hpo_owl,
    load_hpoa_annotations,
    load_hoom_hpo_annotations,
    load_monarch_disease_hpo_annotations,
    load_mondo_metadata,
    load_orphadata_product4_annotations,
    load_ordo_metadata,
)
from mapping_utils import build_orpha_mapping_index
from normalizers import normalize_hpo_id, normalize_owl_local_id
from phenotype_merge import merge_phenotype_annotation_records
from schemas import PatientProfile

"""Build shared artifacts for the project, including disease profiles,
HPO parent relations, information content values, and an example patient profile.
"""


def save_json(data, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def serialize_profile(profile) -> dict:
    row = asdict(profile)

    if hasattr(profile, "hpo_terms"):
        row["hpo_terms"] = sorted(profile.hpo_terms)

    if hasattr(profile, "propagated_hpo_terms"):
        row["propagated_hpo_terms"] = sorted(profile.propagated_hpo_terms)

    if hasattr(profile, "negative_hpo_terms"):
        row["negative_hpo_terms"] = sorted(profile.negative_hpo_terms)

    return row


def serialize_profiles(disease_profiles: dict) -> dict:
    return {
        disease_id: serialize_profile(profile)
        for disease_id, profile in disease_profiles.items()
    }


def build_patient_profile(
    patient_id: str,
    raw_text: str,
    hpo_terms: list[str],
    hpo_labels: dict[str, str],
    hpo_ancestors: dict[str, set[str]],
) -> PatientProfile:
    normalized_terms = {
        term
        for term in (normalize_hpo_id(x) for x in hpo_terms)
        if term is not None and term in hpo_labels
    }

    propagated_terms = propagate_hpo_terms(normalized_terms, hpo_ancestors)

    return PatientProfile(
        patient_id=patient_id,
        raw_text=raw_text,
        hpo_terms=normalized_terms,
        propagated_hpo_terms=propagated_terms,
    )


def is_valid_disease_profile(profile) -> bool:
    has_hpo = bool(profile.hpo_terms)
    has_description = bool(
        profile.merged_description and profile.merged_description.strip()
    )
    return has_hpo or has_description


def filter_disease_profiles(disease_profiles: dict) -> tuple[dict, dict]:
    filtered = {
        disease_id: profile
        for disease_id, profile in disease_profiles.items()
        if is_valid_disease_profile(profile)
    }

    stats = {
        "total_before_filter": len(disease_profiles),
        "total_after_filter": len(filtered),
        "removed_no_hpo_no_description": len(disease_profiles) - len(filtered),
    }
    return filtered, stats


def load_and_report(label: str, loader: Callable, *args):
    print(f"Loading {label}...")
    data = loader(*args)
    if hasattr(data, "__len__"):
        print(f"{label} loaded: {len(data)}")
    return data


def print_filter_stats(name: str, stats: dict) -> None:
    print(
        f"{name} before filter: {stats['total_before_filter']} | "
        f"after: {stats['total_after_filter']}"
    )


def main() -> None:
    print("Loading HPO...")
    hpo_labels, hpo_parents = load_hpo_owl(HPO_PATH)

    print("Computing HPO ancestors...")
    hpo_ancestors = compute_ancestors(hpo_parents)

    hpoa_records = load_and_report(
        "HPOA annotations",
        load_hpoa_annotations,
        HPOA_PATH,
    )
    hoom_hpo_records = load_and_report(
        "HOOM HPO annotations",
        load_hoom_hpo_annotations,
        HOOM_PATH,
    )
    orphadata_records = load_and_report(
        "Orphadata Product 4 annotations",
        load_orphadata_product4_annotations,
        ORPHADATA_PRODUCT4_PATH,
    )
    monarch_records = load_and_report(
        "Monarch disease-HPO annotations",
        load_monarch_disease_hpo_annotations,
        MONARCH_DISEASE_TO_HPO_PATH,
    )

    raw_phenotype_annotation_records = (
        hpoa_records
        + hoom_hpo_records
        + orphadata_records
        + monarch_records
    )
    print(
        "Total raw phenotype annotation records: "
        f"{len(raw_phenotype_annotation_records)}"
    )

    (
        phenotype_annotation_records,
        term_provenance_by_disease,
        negative_terms_by_disease,
    ) = merge_phenotype_annotation_records(raw_phenotype_annotation_records)

    print(
        "Total deduplicated positive phenotype annotation records: "
        f"{len(phenotype_annotation_records)}"
    )
    print(
        "Diseases with at least one negative phenotype assertion: "
        f"{len(negative_terms_by_disease)}"
    )

    ordo_metadata = load_and_report(
        "ORDO metadata",
        load_ordo_metadata,
        ORDO_PATH,
        normalize_owl_local_id,
    )
    mondo_metadata = load_and_report(
        "MONDO metadata",
        load_mondo_metadata,
        MONDO_PATH,
        normalize_owl_local_id,
    )

    print("Building OMIM/MONDO -> ORPHA mapping index...")
    mapping_index = build_orpha_mapping_index(
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata={},
    )
    print(f"Mapping index size: {len(mapping_index)}")

    print("Building canonical disease profiles...")
    canonical_profiles, alias_to_canonical = build_canonical_disease_profiles(
        phenotype_annotation_records=phenotype_annotation_records,
        term_provenance_by_disease=term_provenance_by_disease,
        negative_terms_by_disease=negative_terms_by_disease,
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata={},
        mapping_index=mapping_index,
        apply_true_path_rule=APPLY_TRUE_PATH_RULE,
    )

    print("Expanding profiles to all mapped aliases...")
    expanded_profiles = expand_alias_profiles(
        canonical_profiles=canonical_profiles,
        alias_to_canonical=alias_to_canonical,
    )

    print("Filtering canonical profiles...")
    canonical_profiles, canonical_filter_stats = filter_disease_profiles(
        canonical_profiles
    )

    print("Filtering expanded alias profiles...")
    expanded_profiles, expanded_filter_stats = filter_disease_profiles(
        expanded_profiles
    )

    print_filter_stats("Canonical profiles", canonical_filter_stats)
    print_filter_stats("Expanded alias profiles", expanded_filter_stats)

    print("Computing term frequencies and IC from canonical profiles...")
    term_frequencies = compute_term_frequencies(canonical_profiles)
    ic_values = compute_information_content(
        term_frequencies=term_frequencies,
        total_diseases=len(canonical_profiles),
    )

    print("Building example patient profile...")
    patient = build_patient_profile(
        patient_id=EXAMPLE_PATIENT["patient_id"],
        raw_text=EXAMPLE_PATIENT["raw_text"],
        hpo_terms=EXAMPLE_PATIENT["hpo_terms"],
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
    )

    annotation_source_counts = {
        "HPOA": len(hpoa_records),
        "HOOM": len(hoom_hpo_records),
        "ORPHADATA_PRODUCT4": len(orphadata_records),
        "MONARCH": len(monarch_records),
        "TOTAL_RAW": len(raw_phenotype_annotation_records),
        "TOTAL_DEDUPLICATED_POSITIVE": len(phenotype_annotation_records),
        "DISEASES_WITH_NEGATIVE_ASSERTIONS": len(negative_terms_by_disease),
    }

    print("Saving outputs...")
    outputs = {
        "canonical_disease_profiles.json": serialize_profiles(canonical_profiles),
        "disease_profiles.json": serialize_profiles(expanded_profiles),
        "hpo_labels.json": hpo_labels,
        "hpo_parents.json": {
        k: sorted(v) for k, v in hpo_parents.items()
        },
        "hpo_ancestors.json": {
            k: sorted(v) for k, v in hpo_ancestors.items()
        },
        "term_frequencies.json": term_frequencies,
        "information_content.json": ic_values,
        "example_patient.json": serialize_profile(patient),
        "orpha_mapping_index.json": mapping_index,
        "alias_to_canonical.json": alias_to_canonical,
        "canonical_filter_stats.json": canonical_filter_stats,
        "expanded_filter_stats.json": expanded_filter_stats,
        "annotation_source_counts.json": annotation_source_counts,
        "term_provenance.json": term_provenance_by_disease,
        "negative_terms_by_disease.json": {
            disease_id: sorted(terms)
            for disease_id, terms in negative_terms_by_disease.items()
        },
    }

    for filename, data in outputs.items():
        save_json(data, OUTPUT_DIR / filename)

    print("Done.")
    print(f"Canonical profiles saved: {len(canonical_profiles)}")
    print(f"Expanded alias profiles saved: {len(expanded_profiles)}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
