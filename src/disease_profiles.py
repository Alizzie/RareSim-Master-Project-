from typing import Dict, Optional

from hpo_utils import propagate_hpo_terms
from mapping_utils import (
    choose_preferred_label,
    merge_source_ids,
    resolve_to_orpha,
)
from normalizers import normalize_disease_id, normalize_hpo_id
from schemas import DiseaseProfile
'''Functions to build comprehensive disease profiles by integrating data from multiple sources (HPOA, ORDO, MONDO, HOOM) and applying propagation rules.'''

def _pick_merged_description(
    ordo_description: Optional[str],
    mondo_description: Optional[str],
    hoom_description: Optional[str],
) -> Optional[str]:
    for desc in (ordo_description, mondo_description, hoom_description):
        if desc and desc.strip():
            return desc.strip()
    return None


def _get_or_create_profile(
    profiles: Dict[str, DiseaseProfile],
    disease_id: str,
    label: Optional[str] = None,
) -> DiseaseProfile:
    if disease_id not in profiles:
        profiles[disease_id] = DiseaseProfile(
            disease_id=disease_id,
            label=label or disease_id,
        )
    elif label:
        profiles[disease_id].label = choose_preferred_label(
            profiles[disease_id].label,
            label,
        )
    return profiles[disease_id]


def build_disease_profiles(
    hpoa_records: list[dict],
    hpo_labels: Dict[str, str],
    hpo_ancestors: Dict[str, set],
    ordo_metadata: Dict[str, dict],
    mondo_metadata: Dict[str, dict],
    hoom_metadata: Dict[str, dict],
    mapping_index: Dict[str, str],
    apply_true_path_rule: bool = True,
) -> Dict[str, DiseaseProfile]:
    profiles: Dict[str, DiseaseProfile] = {}

    # HPOA phenotype annotations
    for record in hpoa_records:
        raw_disease_id = normalize_disease_id(record["database_id"])
        hpo_id = normalize_hpo_id(record["hpo_id"])

        if raw_disease_id is None or hpo_id is None:
            continue
        if hpo_id not in hpo_labels:
            continue

        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=None,
        )

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=record["disease_name"],
        )
        profile.hpo_terms.add(hpo_id)
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "hpoa_original_id",
            record["database_id"],
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # ORDO metadata
    for local_id, meta in ordo_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.ordo_label = meta.get("label")
        profile.ordo_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "ordo_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # MONDO metadata
    for local_id, meta in mondo_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.mondo_label = meta.get("label")
        profile.mondo_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "mondo_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # HOOM metadata
    for local_id, meta in hoom_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.hoom_label = meta.get("label")
        profile.hoom_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "hoom_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # Finalize propagated terms and merged description
    for profile in profiles.values():
        if apply_true_path_rule:
            profile.propagated_hpo_terms = propagate_hpo_terms(
                profile.hpo_terms,
                hpo_ancestors,
            )
        else:
            profile.propagated_hpo_terms = set(profile.hpo_terms)

        profile.merged_description = _pick_merged_description(
            profile.ordo_description,
            profile.mondo_description,
            profile.hoom_description,
        )

    return profiles
