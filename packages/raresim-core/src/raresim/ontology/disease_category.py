"""Utilities for ORDO disease categories and hierarchy paths."""

from raresim.utils.normalizers import normalize_disease_id


def collect_matched_aliases(disease_id: str, profile: dict) -> list[str]:
    """Collect equivalent disease IDs from a disease profile."""
    aliases = set()

    normalized_disease_id = normalize_disease_id(disease_id)
    if normalized_disease_id:
        aliases.add(normalized_disease_id)

    source_ids = profile.get("source_ids", {})
    if isinstance(source_ids, dict):
        for value in source_ids.values():
            if isinstance(value, str):
                normalized = normalize_disease_id(value)
                if normalized:
                    aliases.add(normalized)
            elif isinstance(value, (list, set, tuple)):
                for item in value:
                    normalized = normalize_disease_id(item)
                    if normalized:
                        aliases.add(normalized)

    stored_aliases = profile.get("aliases", [])
    if isinstance(stored_aliases, (list, set, tuple)):
        for alias in stored_aliases:
            normalized = normalize_disease_id(alias)
            if normalized:
                aliases.add(normalized)

    return sorted(aliases)


def resolve_category_source_id(
    disease_id: str,
    matched_aliases: list[str],
    disease_ancestors: dict[str, list[str]],
) -> str | None:
    """
    Choose the ID used to build an ORDO category path.

    Preference:
    1. disease_id itself, if it has ORDO ancestors
    2. any ORPHA alias with ORDO ancestors
    3. any other alias with ORDO ancestors
    4. None
    """
    if disease_id in disease_ancestors:
        return disease_id

    for alias in matched_aliases:
        if alias.startswith("ORPHA:") and alias in disease_ancestors:
            return alias

    for alias in matched_aliases:
        if alias in disease_ancestors:
            return alias

    return None


def format_profile_type(profile_type: str | None) -> str | None:
    """Return a readable profile type label."""
    if not profile_type:
        return None

    labels = {
        "specific_disease": "Specific disease",
        "disease_group": "Disease group",
        "category": "Category",
        "subtype": "Subtype",
        "clinical_subtype": "Clinical subtype",
        "etiological_subtype": "Etiological subtype",
        "histopathological_subtype": "Histopathological subtype",
    }
    return labels.get(profile_type, profile_type.replace("_", " ").title())


def build_category_path(
    disease_id: str,
    disease_ancestors: dict[str, list[str]],
    disease_metadata_index: dict[str, dict],
    max_depth: int = 5,
) -> list[dict]:
    """
    Build a readable ORDO category path for a disease.

    disease_ancestors is expected to be ordered root -> immediate parent.
    """
    ancestor_ids = disease_ancestors.get(disease_id, [])
    ancestor_ids = ancestor_ids[-max_depth:]

    path = []

    for ancestor_id in ancestor_ids:
        meta = disease_metadata_index.get(ancestor_id, {})
        profile_type = meta.get("profile_type")

        path.append(
            {
                "disease_id": ancestor_id,
                "label": meta.get("label") or ancestor_id,
                "profile_type": profile_type,
                "profile_type_label": format_profile_type(profile_type),
            }
        )

    return path


def build_category_metadata(
    disease_id: str,
    profile: dict,
    disease_ancestors: dict[str, list[str]],
    disease_metadata_index: dict[str, dict],
) -> dict:
    """
    Build all category-related metadata for a ranked disease result.

    This is the helper pipelines should use before creating SimilarityResult.
    """
    matched_aliases = collect_matched_aliases(disease_id, profile)

    category_source_id = resolve_category_source_id(
        disease_id=disease_id,
        matched_aliases=matched_aliases,
        disease_ancestors=disease_ancestors,
    )

    category_path = (
        build_category_path(
            disease_id=category_source_id,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
        )
        if category_source_id
        else []
    )

    category_source_metadata = (
        disease_metadata_index.get(category_source_id, {})
        if category_source_id
        else {}
    )

    profile_type = (
        profile.get("profile_type")
        or category_source_metadata.get("profile_type")
    )

    return {
        "profile_type": profile_type,
        "category_source_id": category_source_id,
        "category_path": category_path,
        "matched_aliases": matched_aliases,
    }
