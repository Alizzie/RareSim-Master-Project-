"""Module to build disease profiles by integrating data from multiple sources."""

from copy import deepcopy

from raresim.ontology.hpo_utils import propagate_hpo_terms
from raresim.utils.mapping_utils import (
    choose_preferred_label,
    merge_source_ids,
    resolve_to_orpha,
)
from raresim.utils.normalizers import normalize_disease_id, normalize_hpo_id
from raresim.types.schemas import DiseaseProfile

_DESCRIPTION_FIELDS = (
    "ordo_label",
    "ordo_description",
    "mondo_label",
    "mondo_description",
    "hoom_label",
    "hoom_description",
)


def _pick_merged_description(
    ordo_description: str | None,
    mondo_description: str | None,
    hoom_description: str | None,
) -> str | None:
    for desc in (ordo_description, mondo_description, hoom_description):
        if desc and desc.strip():
            return desc.strip()
    return None


def _get_or_create_profile(
    profiles: dict[str, DiseaseProfile],
    disease_id: str,
    label: str | None = None,
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
    profiles: dict[str, DiseaseProfile],
    hpo_ancestors: dict[str, set[str]],
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


def _is_linkable_source_id(key: str, value: object) -> bool:
    """
    Return True only for source_ids values that can safely identify
    the same disease across sources.

    Do not use frequency codes, HPO terms, or arbitrary strings for
    cross-profile matching.
    """
    if not isinstance(value, str):
        return False

    if key.endswith("_frequency_codes"):
        return False

    if value.startswith("HP:"):
        return False

    disease_prefixes = (
        "ORPHA:",
        "OMIM:",
        "MONDO:",
        "DECIPHER:",
        "DOID:",
        "MIM:",
        "Orphanet_",
        "MONDO_",
    )

    return value.startswith(disease_prefixes)


def _build_source_id_index(
    profiles: dict[str, DiseaseProfile],
) -> dict[str, list[str]]:
    """
    Build an inverted index using only safe disease identifiers.

    This avoids false matches caused by non-ID source_ids values such as
    HPOA frequency codes like '5/5' or 'HP:0040283'.
    """
    index: dict[str, list[str]] = {}

    for canonical_key, profile in profiles.items():
        index.setdefault(canonical_key, []).append(canonical_key)

        for key, val in (profile.source_ids or {}).items():
            if _is_linkable_source_id(key, val):
                index.setdefault(val, []).append(canonical_key)

    return index


def _propagate_descriptions(
    profiles: dict[str, DiseaseProfile],
) -> int:
    """
    Final pass: copy description metadata to canonical profiles that
    lack it but represent the same disease as a profile that has it.

    This handles the case where the same disease has two canonical entries:
      - One from ORDO/MONDO metadata  → has description, keyed as ORPHA:NNNN
      - One from Monarch annotations  → no description, keyed as MONDO:NNNNNNN

    Both exist as independent canonical entries (neither is an alias of
    the other in alias_to_canonical). The connection is found by looking
    for shared values in source_ids — ORPHA:102002 stores "mondo_id":
    "MONDO:0000437", and MONDO:0000437 itself is a canonical key, so
    searching the index for "MONDO:0000437" finds both.

    Must be called AFTER _finalize_profiles so merged_description is
    already set on profiles that have source descriptions.

    Returns the number of profiles that received description data.
    """
    # Separate profiles into those that have descriptions and those that do not.
    rich = {
        k: p
        for k, p in profiles.items()
        if p.merged_description and p.merged_description.strip()
    }
    poor = {
        k: p
        for k, p in profiles.items()
        if not (p.merged_description and p.merged_description.strip())
    }

    if not rich or not poor:
        return 0

    # Build the source ID index over rich profiles only — we only want
    # to find rich profiles as candidates, not other poor ones.
    rich_index = _build_source_id_index(rich)

    n_updated = 0

    for poor_key, poor_profile in poor.items():
        candidates: set[str] = set()

        # The poor profile's own canonical key may appear as a source_id
        # value inside a rich profile (e.g. ORPHA:102002 stores mondo_id:
        # MONDO:0000437, so searching for MONDO:0000437 finds ORPHA:102002)
        for match_key in rich_index.get(poor_key, []):
            candidates.add(match_key)

        # Also search by every source_id value of the poor profile itself
        for val in (poor_profile.source_ids or {}).values():
            if isinstance(val, str):
                for match_key in rich_index.get(val, []):
                    candidates.add(match_key)

        # Never match a profile to itself
        candidates.discard(poor_key)

        if not candidates:
            continue

        # Pick the richest candidate — most description fields populated
        best_key = max(
            candidates,
            key=lambda k: sum(
                1 for f in _DESCRIPTION_FIELDS if getattr(profiles[k], f)
            ),
        )
        best = profiles[best_key]

        # Copy missing description fields from best → poor_profile
        changed = False
        for field in _DESCRIPTION_FIELDS:
            src_val = getattr(best, field)
            if src_val and not getattr(poor_profile, field):
                setattr(poor_profile, field, src_val)
                changed = True

        if changed:
            # Recompute merged_description now that source fields are populated
            poor_profile.merged_description = _pick_merged_description(
                poor_profile.ordo_description,
                poor_profile.mondo_description,
                poor_profile.hoom_description,
            )
            n_updated += 1

    return n_updated


def _as_optional_str(value: object) -> str | None:
    """Return a stripped string if value is a non-empty string."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_metadata_lookup(
    ordo_metadata: dict[str, dict],
    mondo_metadata: dict[str, dict],
    hoom_metadata: dict[str, dict],
) -> dict[str, dict]:
    """Build lookup from normalized disease ID to source metadata."""
    metadata_by_normalized_id: dict[str, dict] = {}

    for metadata_source in (ordo_metadata, mondo_metadata, hoom_metadata):
        for meta in metadata_source.values():
            normalized_id = _as_optional_str(meta.get("normalized_id"))
            if normalized_id:
                metadata_by_normalized_id[normalized_id] = meta

    return metadata_by_normalized_id


def _add_negative_terms(
    profile: DiseaseProfile,
    disease_id: str,
    negative_terms_by_disease: dict[str, set[str]],
) -> None:
    """Attach negative HPO terms to a profile if they exist."""
    negative_terms = negative_terms_by_disease.get(disease_id, set())

    if negative_terms:
        profile.negative_hpo_terms.update(negative_terms)


def _mark_canonicalized_if_needed(
    profile: DiseaseProfile,
    canonical_disease_id: str,
    raw_disease_id: str,
) -> None:
    """Mark a profile as canonicalized when its ID changed during mapping."""
    if canonical_disease_id != raw_disease_id:
        profile.canonicalized_to_orpha = True


def _add_annotation_record_to_profile(
    record: dict,
    profiles: dict[str, DiseaseProfile],
    alias_to_canonical: dict[str, str],
    metadata_by_normalized_id: dict[str, dict],
    build_data: dict,
) -> None:
    """Add one phenotype annotation record to the canonical profile collection."""
    raw_disease_id = normalize_disease_id(record["database_id"])
    hpo_id = normalize_hpo_id(record["hpo_id"])

    if raw_disease_id is None or hpo_id is None:
        return

    if hpo_id not in build_data["hpo_labels"]:
        return

    source_meta = metadata_by_normalized_id.get(raw_disease_id)
    canonical_disease_id = resolve_to_orpha(
        raw_disease_id,
        mapping_index=build_data["mapping_index"],
        source_metadata=source_meta,
    )

    alias_to_canonical[raw_disease_id] = canonical_disease_id

    profile = _get_or_create_profile(
        profiles,
        canonical_disease_id,
        label=_as_optional_str(record.get("disease_name")),
    )
    profile.hpo_terms.add(hpo_id)

    source_name = (_as_optional_str(record.get("source")) or "UNKNOWN").lower()
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
            str(freq_code),
        )

    disease_term_prov = build_data["term_provenance_by_disease"].get(
        raw_disease_id,
        {},
    )
    if hpo_id in disease_term_prov:
        profile.term_provenance[hpo_id] = disease_term_prov[hpo_id]

    _add_negative_terms(
        profile,
        raw_disease_id,
        build_data["negative_terms_by_disease"],
    )
    _mark_canonicalized_if_needed(profile, canonical_disease_id, raw_disease_id)


def _apply_source_metadata(
    profile: DiseaseProfile,
    meta: dict,
    source_name: str,
) -> None:
    """Copy source-specific label and description metadata onto a profile."""
    label = _as_optional_str(meta.get("label"))
    description = _as_optional_str(meta.get("description"))

    if source_name == "ORDO":
        profile.ordo_label = label
        profile.ordo_description = description
        profile.profile_type = _as_optional_str(meta.get("profile_type")) or profile.profile_type
    elif source_name == "MONDO":
        profile.mondo_label = label
        profile.mondo_description = description
    elif source_name == "HOOM":
        profile.hoom_label = label
        profile.hoom_description = description


def _add_metadata_entry_to_profile(
    local_id: str,
    meta: dict,
    source_name: str,
    profiles: dict[str, DiseaseProfile],
    build_data: dict,
) -> tuple[str, str]:
    """Add one ORDO, MONDO, or HOOM metadata entry to the profile collection."""
    raw_disease_id = str(meta["normalized_id"])
    canonical_disease_id = resolve_to_orpha(
        raw_disease_id,
        mapping_index=build_data["mapping_index"],
        source_metadata=meta,
    )

    profile = _get_or_create_profile(
        profiles,
        canonical_disease_id,
        label=_as_optional_str(meta.get("label")),
    )

    _apply_source_metadata(profile, meta, source_name)

    source_prefix = source_name.lower()
    profile.source_ids = merge_source_ids(
        profile.source_ids,
        f"{source_prefix}_id",
        raw_disease_id,
    )
    profile.source_ids = merge_source_ids(
        profile.source_ids,
        f"{source_prefix}_local_id",
        local_id,
    )

    _add_negative_terms(
        profile,
        raw_disease_id,
        build_data["negative_terms_by_disease"],
    )
    _mark_canonicalized_if_needed(profile, canonical_disease_id, raw_disease_id)

    return raw_disease_id, canonical_disease_id


def _add_metadata_collection_to_profiles(
    metadata: dict[str, dict],
    source_name: str,
    profiles: dict[str, DiseaseProfile],
    alias_to_canonical: dict[str, str],
    build_data: dict,
) -> None:
    """Add all metadata entries from one source to canonical disease profiles."""
    for local_id, meta in metadata.items():
        raw_disease_id, canonical_disease_id = _add_metadata_entry_to_profile(
            local_id,
            meta,
            source_name,
            profiles,
            build_data,
        )
        alias_to_canonical[raw_disease_id] = canonical_disease_id


def build_canonical_disease_profiles(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    phenotype_annotation_records: list[dict],
    term_provenance_by_disease: dict[str, dict],
    negative_terms_by_disease: dict[str, set[str]],
    hpo_labels: dict[str, str],
    hpo_ancestors: dict[str, set[str]],
    ordo_metadata: dict[str, dict],
    mondo_metadata: dict[str, dict],
    hoom_metadata: dict[str, dict],
    mapping_index: dict[str, str],
    apply_true_path_rule: bool = True,
) -> tuple[dict[str, DiseaseProfile], dict[str, str]]:
    """
    Build canonical disease profiles keyed by canonical disease IDs.

    The canonical ID is preferably an ORPHA ID when a reliable mapping exists.
    The returned alias map stores links from source-specific IDs to canonical IDs.
    """
    profiles: dict[str, DiseaseProfile] = {}
    alias_to_canonical: dict[str, str] = {}

    metadata_by_normalized_id = _build_metadata_lookup(
        ordo_metadata,
        mondo_metadata,
        hoom_metadata,
    )

    build_data = {
        "term_provenance_by_disease": term_provenance_by_disease,
        "negative_terms_by_disease": negative_terms_by_disease,
        "hpo_labels": hpo_labels,
        "mapping_index": mapping_index,
    }

    for record in phenotype_annotation_records:
        _add_annotation_record_to_profile(
            record,
            profiles,
            alias_to_canonical,
            metadata_by_normalized_id,
            build_data,
        )

    _add_metadata_collection_to_profiles(
        ordo_metadata,
        "ORDO",
        profiles,
        alias_to_canonical,
        build_data,
    )
    _add_metadata_collection_to_profiles(
        mondo_metadata,
        "MONDO",
        profiles,
        alias_to_canonical,
        build_data,
    )
    _add_metadata_collection_to_profiles(
        hoom_metadata,
        "HOOM",
        profiles,
        alias_to_canonical,
        build_data,
    )

    _finalize_profiles(
        profiles=profiles,
        hpo_ancestors=hpo_ancestors,
        apply_true_path_rule=apply_true_path_rule,
    )

    return profiles, alias_to_canonical


def expand_alias_profiles(
    canonical_profiles: dict[str, DiseaseProfile],
    alias_to_canonical: dict[str, str],
) -> dict[str, DiseaseProfile]:
    """
    Create additional profiles for aliases that point to canonical profiles.

    Each alias profile is a copy of its canonical profile, but its disease_id is
    replaced by the alias ID and its source_ids include the canonical ID.
    """
    expanded_profiles: dict[str, DiseaseProfile] = {}

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
