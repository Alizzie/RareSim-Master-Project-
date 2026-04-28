from copy import deepcopy
from typing import Dict, Optional, Set, Tuple

from core.hpo_utils import propagate_hpo_terms
from core.mapping_utils import (
    choose_preferred_label,
    merge_source_ids,
    resolve_to_orpha,
)
from core.normalizers import normalize_disease_id, normalize_hpo_id
from core.schemas import DiseaseProfile

"""Module to build disease profiles by integrating data from multiple sources."""


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
    hpo_ancestors: Dict[str, Set[str]],
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
    phenotype_annotation_records: list[dict],
    term_provenance_by_disease: Dict[str, dict],
    negative_terms_by_disease: Dict[str, Set[str]],
    hpo_labels: Dict[str, str],
    hpo_ancestors: Dict[str, Set[str]],
    ordo_metadata: Dict[str, dict],
    mondo_metadata: Dict[str, dict],
    hoom_metadata: Dict[str, dict],
    mapping_index: Dict[str, str],
    apply_true_path_rule: bool = True,
) -> Tuple[Dict[str, DiseaseProfile], Dict[str, str]]:
    """
    Build canonical profiles keyed by canonical disease IDs
    (preferably ORPHA when mapping exists).
    """
    profiles: Dict[str, DiseaseProfile] = {}
    alias_to_canonical: Dict[str, str] = {}

    # 1. merged disease -> phenotype annotations
    for record in phenotype_annotation_records:
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
            label=record.get("disease_name"),
        )
        profile.hpo_terms.add(hpo_id)

        source_name = (record.get("source") or "UNKNOWN").lower()
        profile.source_ids = merge_source_ids(
            profile.source_ids,
            f"{source_name}_original_id",
            raw_disease_id,
        )

        freq_code = record.get("frequency_code")
        if freq_code:
            profile.source_ids = merge_source_ids(
                profile.source_ids,
                f"{source_name}_frequency_codes",
                freq_code,
            )

        disease_term_prov = term_provenance_by_disease.get(raw_disease_id, {})
        if hpo_id in disease_term_prov:
            profile.term_provenance[hpo_id] = disease_term_prov[hpo_id]

        negative_terms = negative_terms_by_disease.get(raw_disease_id, set())
        if negative_terms:
            profile.negative_hpo_terms.update(negative_terms)

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

        negative_terms = negative_terms_by_disease.get(raw_disease_id, set())
        if negative_terms:
            profile.negative_hpo_terms.update(negative_terms)

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

        negative_terms = negative_terms_by_disease.get(raw_disease_id, set())
        if negative_terms:
            profile.negative_hpo_terms.update(negative_terms)

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

        negative_terms = negative_terms_by_disease.get(raw_disease_id, set())
        if negative_terms:
            profile.negative_hpo_terms.update(negative_terms)

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
    expanded_profiles: Dict[str, DiseaseProfile] = {}

    for canonical_id, profile in canonical_profiles.items():
        expanded_profiles[canonical_id] = deepcopy(profile)

    for alias_id, canonical_id in alias_to_canonical.items():
        if canonical_id not in canonical_profiles:
            continue

        canonical_profile = canonical_profiles[canonical_id]

        alias_profile = deepcopy(canonical_profile)
        alias_profile.disease_id = alias_id

        alias_profile.source_ids = dict(alias_profile.source_ids)
        alias_profile.source_ids["canonical_id"] = canonical_id
        alias_profile.hpo_terms = set(canonical_profile.hpo_terms)
        alias_profile.propagated_hpo_terms = set(canonical_profile.propagated_hpo_terms)
        alias_profile.term_provenance = dict(canonical_profile.term_provenance)
        alias_profile.negative_hpo_terms = set(canonical_profile.negative_hpo_terms)

        expanded_profiles[alias_id] = alias_profile

    mondo_with_hpo = sum(
        1
        for did, profile in expanded_profiles.items()
        if did.startswith("MONDO:") and len(profile.hpo_terms) > 0
    )
    print("MONDO entries with HPO after propagation:", mondo_with_hpo)

    return expanded_profiles
