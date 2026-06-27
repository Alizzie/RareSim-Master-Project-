"""Build shared artifacts for the project, including disease profiles,
HPO parent relations, information content values, and an example patient profile.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from raresim.utils.io import save_json

from raresim.core.config import (
    APPLY_TRUE_PATH_RULE,
    EXAMPLE_PATIENT,
)
from raresim.utils.paths import (
    HPO_PATH,
    HPOA_PATH,
    HOOM_PATH,
    MONARCH_DISEASE_TO_HPO_PATH,
    MONDO_PATH,
    ORDO_PATH,
    ORPHADATA_PRODUCT4_PATH,
    ARTIFACTS_DIR,
)
from raresim.ontology.disease_profiles import (
    build_canonical_disease_profiles,
    expand_alias_profiles,
)
from raresim.ontology.hpo_utils import compute_ancestors, propagate_hpo_terms
from raresim.ontology.ic import compute_information_content, compute_term_frequencies
from raresim.ontology.loaders import (
    load_hpo_owl,
    load_hpoa_annotations,
    load_hoom_hpo_annotations,
    load_monarch_disease_hpo_annotations,
    load_mondo_metadata,
    load_orphadata_product4_annotations,
    load_ordo_metadata,
    load_ordo_parents,
)
from raresim.utils.mapping_utils import build_orpha_mapping_index
from raresim.utils.normalizers import normalize_hpo_id, normalize_owl_local_id
from raresim.ontology.phenotype_merge import merge_phenotype_annotation_records
from raresim.ontology.disease_ancestors import build_ordered_ancestor_chains
from raresim.types.schemas import PatientProfile


@dataclass
class AnnotationArtifacts: # pylint: disable=too-many-instance-attributes
    """Container for loaded and merged phenotype annotation records."""

    hpoa_records: list[dict]
    hoom_hpo_records: list[dict]
    orphadata_records: list[dict]
    monarch_records: list[dict]
    raw_records: list[dict]
    phenotype_records: list[dict]
    term_provenance_by_disease: dict
    negative_terms_by_disease: dict


@dataclass
class MetadataArtifacts:
    """Container for loaded disease metadata and mapping indexes."""

    ordo_metadata: dict
    mondo_metadata: dict
    hoom_metadata: dict
    disease_metadata_index: dict
    mapping_index: dict


@dataclass
class ProfileArtifacts:
    """Container for canonical and expanded disease profile outputs."""

    canonical_profiles: dict
    expanded_profiles: dict
    alias_to_canonical: dict
    canonical_filter_stats: dict
    expanded_filter_stats: dict


def make_json_safe(value: object) -> object:
    """Recursively convert Python objects into JSON-serializable values."""
    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    return value


def serialize_profile(profile) -> dict[str, object]:
    """Convert a profile dataclass into a JSON-serializable dictionary."""
    row = make_json_safe(asdict(profile))

    if not isinstance(row, dict):
        raise TypeError(
            f"Expected serialized profile to be a dict, got {type(row).__name__}"
        )

    return row


def serialize_profiles(disease_profiles: dict) -> dict:
    """Convert a dictionary of disease profiles into a JSON-serializable format."""
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
    """Build an example patient profile with normalized and propagated HPO terms."""
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
    """Return True if a disease profile has HPO terms or a description."""
    has_hpo = bool(profile.hpo_terms)
    has_description = bool(
        profile.merged_description and profile.merged_description.strip()
    )
    return has_hpo or has_description


def filter_disease_profiles(disease_profiles: dict) -> tuple[dict, dict]:
    """Remove profiles that have neither HPO terms nor descriptions."""
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


def build_disease_metadata_index(ordo_metadata: dict) -> dict:
    """Build ORPHA ID -> display metadata for ORDO diseases/categories."""
    index = {}

    for meta in ordo_metadata.values():
        disease_id = meta.get("normalized_id")
        if not disease_id:
            continue

        index[disease_id] = {
            "label": meta.get("label") or disease_id,
            "profile_type": meta.get("profile_type"),
        }

    return index


def load_and_report(label: str, loader: Callable, *args):
    """Load data with a loader function and print a short status report."""
    print(f"Loading {label}...")
    data = loader(*args)
    if hasattr(data, "__len__"):
        print(f"{label} loaded: {len(data)}")
    return data


def print_filter_stats(name: str, stats: dict) -> None:
    """Print before/after filtering statistics for disease profiles."""
    print(
        f"{name} before filter: {stats['total_before_filter']} | "
        f"after: {stats['total_after_filter']}"
    )


def load_hpo_artifacts() -> tuple[dict, dict, dict]:
    """Load HPO labels, HPO parent relations, and computed ancestors."""
    print("Loading HPO...")
    hpo_labels, hpo_parents = load_hpo_owl(HPO_PATH)

    print("Computing HPO ancestors...")
    hpo_ancestors = compute_ancestors(hpo_parents)

    return hpo_labels, hpo_parents, hpo_ancestors


def load_annotation_artifacts() -> AnnotationArtifacts:
    """Load and merge all phenotype annotation sources."""
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

    raw_records = (
        hpoa_records + hoom_hpo_records + orphadata_records + monarch_records
    )
    print(f"Total raw phenotype annotation records: {len(raw_records)}")

    (
        phenotype_records,
        term_provenance_by_disease,
        negative_terms_by_disease,
    ) = merge_phenotype_annotation_records(raw_records)

    print(
        "Total deduplicated positive phenotype annotation records: "
        f"{len(phenotype_records)}"
    )
    print(
        "Diseases with at least one negative phenotype assertion: "
        f"{len(negative_terms_by_disease)}"
    )

    return AnnotationArtifacts(
        hpoa_records=hpoa_records,
        hoom_hpo_records=hoom_hpo_records,
        orphadata_records=orphadata_records,
        monarch_records=monarch_records,
        raw_records=raw_records,
        phenotype_records=phenotype_records,
        term_provenance_by_disease=term_provenance_by_disease,
        negative_terms_by_disease=negative_terms_by_disease,
    )


def load_metadata_artifacts() -> MetadataArtifacts:
    """Load disease metadata and build mapping/index artifacts."""
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

    hoom_metadata = {}
    disease_metadata_index = build_disease_metadata_index(ordo_metadata)

    print("Building OMIM/MONDO -> ORPHA mapping index...")
    mapping_index = build_orpha_mapping_index(
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata=hoom_metadata,
    )
    print(f"Mapping index size: {len(mapping_index)}")

    return MetadataArtifacts(
        ordo_metadata=ordo_metadata,
        mondo_metadata=mondo_metadata,
        hoom_metadata=hoom_metadata,
        disease_metadata_index=disease_metadata_index,
        mapping_index=mapping_index,
    )


def build_profile_artifacts(
    annotations: AnnotationArtifacts,
    metadata: MetadataArtifacts,
    hpo_labels: dict,
    hpo_ancestors: dict,
) -> ProfileArtifacts:
    """Build, expand, and filter canonical disease profiles."""
    print("Building canonical disease profiles...")
    canonical_profiles, alias_to_canonical = build_canonical_disease_profiles(
        phenotype_annotation_records=annotations.phenotype_records,
        term_provenance_by_disease=annotations.term_provenance_by_disease,
        negative_terms_by_disease=annotations.negative_terms_by_disease,
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
        ordo_metadata=metadata.ordo_metadata,
        mondo_metadata=metadata.mondo_metadata,
        hoom_metadata=metadata.hoom_metadata,
        mapping_index=metadata.mapping_index,
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

    return ProfileArtifacts(
        canonical_profiles=canonical_profiles,
        expanded_profiles=expanded_profiles,
        alias_to_canonical=alias_to_canonical,
        canonical_filter_stats=canonical_filter_stats,
        expanded_filter_stats=expanded_filter_stats,
    )


def build_disease_hierarchy_artifacts() -> tuple[dict, dict]:
    """Load ORDO parent relations and compute disease ancestor chains."""
    print("Extracting ORDO parent relations...")
    disease_parents = load_ordo_parents(ORDO_PATH)
    print(f"  ORDO parent relations: {len(disease_parents)} nodes")

    print("Computing disease ancestors...")
    disease_ancestors = build_ordered_ancestor_chains(disease_parents)
    print(f"  Ancestor chains: {len(disease_ancestors)} diseases")

    return disease_parents, disease_ancestors


def build_annotation_source_counts(annotations: AnnotationArtifacts) -> dict:
    """Build summary counts for annotation sources."""
    return {
        "HPOA": len(annotations.hpoa_records),
        "HOOM": len(annotations.hoom_hpo_records),
        "ORPHADATA_PRODUCT4": len(annotations.orphadata_records),
        "MONARCH": len(annotations.monarch_records),
        "TOTAL_RAW": len(annotations.raw_records),
        "TOTAL_DEDUPLICATED_POSITIVE": len(annotations.phenotype_records),
        "DISEASES_WITH_NEGATIVE_ASSERTIONS": len(
            annotations.negative_terms_by_disease
        ),
    }


def main() -> None:
    """Build and save all shared JSON artifacts used by the project."""
    print("Download ontologies")

    hpo_labels, hpo_parents, hpo_ancestors = load_hpo_artifacts()
    annotations = load_annotation_artifacts()
    metadata = load_metadata_artifacts()

    profiles = build_profile_artifacts(
        annotations=annotations,
        metadata=metadata,
        hpo_labels=hpo_labels,
        hpo_ancestors=hpo_ancestors,
    )

    disease_parents, disease_ancestors = build_disease_hierarchy_artifacts()

    print("Computing term frequencies and IC from canonical profiles...")
    term_frequencies = compute_term_frequencies(profiles.canonical_profiles)
    ic_values = compute_information_content(
        term_frequencies=term_frequencies,
        total_diseases=len(profiles.canonical_profiles),
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
    outputs = {
        "canonical_disease_profiles.json": serialize_profiles(
            profiles.canonical_profiles
        ),
        "disease_profiles.json": serialize_profiles(profiles.expanded_profiles),
        "hpo_labels.json": hpo_labels,
        "hpo_parents.json": {k: sorted(v) for k, v in hpo_parents.items()},
        "hpo_ancestors.json": {k: sorted(v) for k, v in hpo_ancestors.items()},
        "disease_parents.json": {k: sorted(v) for k, v in disease_parents.items()},
        "disease_ancestors.json": disease_ancestors,
        "disease_metadata_index.json": metadata.disease_metadata_index,
        "term_frequencies.json": term_frequencies,
        "information_content.json": ic_values,
        "example_patient.json": serialize_profile(patient),
        "orpha_mapping_index.json": metadata.mapping_index,
        "alias_to_canonical.json": profiles.alias_to_canonical,
        "canonical_filter_stats.json": profiles.canonical_filter_stats,
        "expanded_filter_stats.json": profiles.expanded_filter_stats,
        "annotation_source_counts.json": build_annotation_source_counts(annotations),
        "term_provenance.json": annotations.term_provenance_by_disease,
        "negative_terms_by_disease.json": {
            disease_id: sorted(terms)
            for disease_id, terms in annotations.negative_terms_by_disease.items()
        },
    }

    for filename, data in outputs.items():
        save_json(data, ARTIFACTS_DIR / filename)

    print("Done.")
    print(f"Canonical profiles saved: {len(profiles.canonical_profiles)}")
    print(f"Expanded alias profiles saved: {len(profiles.expanded_profiles)}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
