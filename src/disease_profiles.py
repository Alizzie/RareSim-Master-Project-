from copy import deepcopy
from typing import Dict, Optional, Tuple

from hpo_utils import propagate_hpo_terms
from mapping_utils import (
    choose_preferred_label,
    merge_source_ids,
    resolve_to_orpha,
)
from normalizers import normalize_disease_id, normalize_hpo_id
from schemas import DiseaseProfile
'''Module to build disease profiles by integrating data from multiple sources:'''

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


def _finalize_profiles(
    profiles: Dict[str, DiseaseProfile],
    hpo_ancestors: Dict[str, set],
    apply_true_path_rule: bool,
) -> None:
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


def build_canonical_disease_profiles(
    hpoa_records: list[dict],
    hpo_labels: Dict[str, str],
    hpo_ancestors: Dict[str, set],
    ordo_metadata: Dict[str, dict],
    mondo_metadata: Dict[str, dict],
    hoom_metadata: Dict[str, dict],
    mapping_index: Dict[str, str],
    apply_true_path_rule: bool = True,
) -> Tuple[Dict[str, DiseaseProfile], Dict[str, str]]:
    """
    Build canonical profiles keyed by canonical disease IDs
    (preferably ORPHA when mapping exists).

    Returns:
        canonical_profiles: canonical_id -> DiseaseProfile
        alias_to_canonical: alias_id -> canonical_id
    """
    profiles: Dict[str, DiseaseProfile] = {}
    alias_to_canonical: Dict[str, str] = {}

    # 1. HPOA disease -> phenotype annotations
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

        alias_to_canonical[raw_disease_id] = canonical_disease_id

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=record["disease_name"],
        )
        profile.hpo_terms.add(hpo_id)
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "hpoa_original_id",
            raw_disease_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # 2. ORDO metadata
    for local_id, meta in ordo_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        alias_to_canonical[raw_disease_id] = canonical_disease_id

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.ordo_label = meta.get("label")
        profile.ordo_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "ordo_id",
            raw_disease_id,
        )
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "ordo_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # 3. MONDO metadata
    for local_id, meta in mondo_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        alias_to_canonical[raw_disease_id] = canonical_disease_id

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.mondo_label = meta.get("label")
        profile.mondo_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "mondo_id",
            raw_disease_id,
        )
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "mondo_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    # 4. HOOM metadata
    for local_id, meta in hoom_metadata.items():
        raw_disease_id = meta["normalized_id"]
        canonical_disease_id = resolve_to_orpha(
            raw_disease_id,
            mapping_index=mapping_index,
            source_metadata=meta,
        )

        alias_to_canonical[raw_disease_id] = canonical_disease_id

        profile = _get_or_create_profile(
            profiles,
            canonical_disease_id,
            label=meta.get("label"),
        )
        profile.hoom_label = meta.get("label")
        profile.hoom_description = meta.get("description")
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "hoom_id",
            raw_disease_id,
        )
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            "hoom_local_id",
            local_id,
        )

        if canonical_disease_id != raw_disease_id:
            profile.canonicalized_to_orpha = True

    _finalize_profiles(
        profiles=profiles,
        hpo_ancestors=hpo_ancestors,
        apply_true_path_rule=apply_true_path_rule,
    )

    return profiles, alias_to_canonical


def expand_alias_profiles(
    canonical_profiles: Dict[str, DiseaseProfile],
    alias_to_canonical: Dict[str, str],
) -> Dict[str, DiseaseProfile]:
    """
    Expand canonical profiles to all aliases AND force HPO propagation.

    After this:
    - If a MONDO / OMIM / ORPHA alias maps to a canonical disease,
      it will inherit its HPO terms.
    """
    expanded_profiles: Dict[str, DiseaseProfile] = {}

    # 1. keep canonical profiles
    for canonical_id, profile in canonical_profiles.items():
        expanded_profiles[canonical_id] = deepcopy(profile)

    # 2. propagate to aliases
    for alias_id, canonical_id in alias_to_canonical.items():
        if canonical_id not in canonical_profiles:
            continue

        canonical_profile = canonical_profiles[canonical_id]

        # create alias profile
        alias_profile = deepcopy(canonical_profile)
        alias_profile.disease_id = alias_id

        # track provenance
        alias_profile.source_ids = dict(alias_profile.source_ids)
        alias_profile.source_ids["canonical_id"] = canonical_id

        # FORCE HPO PROPAGATION
        # even if alias originally had no HPO terms
        alias_profile.hpo_terms = set(canonical_profile.hpo_terms)
        alias_profile.propagated_hpo_terms = set(
            canonical_profile.propagated_hpo_terms
        )

        expanded_profiles[alias_id] = alias_profile
        mondo_with_hpo = sum(
        1 for did, p in expanded_profiles.items()
        if did.startswith("MONDO:") and len(p.hpo_terms) > 0)

    print("MONDO entries with HPO after propagation:", mondo_with_hpo)

    return expanded_profiles

