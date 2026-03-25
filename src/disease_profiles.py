from typing import Dict, Optional

from hpo_utils import propagate_hpo_terms
from normalizers import (
    normalize_disease_id,
    normalize_hpo_id,
    normalize_owl_local_id,
)
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


def build_disease_profiles(
    hpoa_records: list[dict],
    hpo_labels: Dict[str, str],
    hpo_ancestors: Dict[str, set],
    ordo_metadata: Dict[str, dict],
    mondo_metadata: Dict[str, dict],
    hoom_metadata: Dict[str, dict],
    apply_true_path_rule: bool = True,
) -> Dict[str, DiseaseProfile]:
    profiles: Dict[str, DiseaseProfile] = {}

    # Step 1: start from HPOA disease -> phenotype annotations
    for record in hpoa_records:
        disease_id = normalize_disease_id(record["database_id"])
        hpo_id = normalize_hpo_id(record["hpo_id"])

        if disease_id is None or hpo_id is None:
            continue
        if hpo_id not in hpo_labels:
            continue

        if disease_id not in profiles:
            profiles[disease_id] = DiseaseProfile(
                disease_id=disease_id,
                label=record["disease_name"],
                source_ids={"original_id": record["database_id"]},
            )

        profiles[disease_id].hpo_terms.add(hpo_id)

    # Step 2: enrich with ORDO metadata
    for local_id, meta in ordo_metadata.items():
        disease_id = normalize_owl_local_id(local_id)

        if disease_id not in profiles:
            profiles[disease_id] = DiseaseProfile(
                disease_id=disease_id,
                label=meta.get("label") or disease_id,
            )

        profiles[disease_id].ordo_label = meta.get("label")
        profiles[disease_id].ordo_description = meta.get("description")

    # Step 3: enrich with MONDO metadata
    for local_id, meta in mondo_metadata.items():
        disease_id = normalize_owl_local_id(local_id)

        if disease_id not in profiles:
            profiles[disease_id] = DiseaseProfile(
                disease_id=disease_id,
                label=meta.get("label") or disease_id,
            )

        profiles[disease_id].mondo_label = meta.get("label")
        profiles[disease_id].mondo_description = meta.get("description")

    # Step 4: enrich with HOOM metadata
    for local_id, meta in hoom_metadata.items():
        disease_id = normalize_owl_local_id(local_id)

        if disease_id not in profiles:
            profiles[disease_id] = DiseaseProfile(
                disease_id=disease_id,
                label=meta.get("label") or disease_id,
            )

        profiles[disease_id].hoom_label = meta.get("label")
        profiles[disease_id].hoom_description = meta.get("description")

    # Step 5: apply propagation and merged description
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
