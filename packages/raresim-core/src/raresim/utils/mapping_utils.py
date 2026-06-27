"""Utility functions for mapping disease identifiers across different ontologies and sources."""

import re

ORPHA_PATTERNS = [
    re.compile(r"^ORPHA:(\d+)$", re.IGNORECASE),
    re.compile(r"^ORPHANET:(\d+)$", re.IGNORECASE),
    re.compile(r"^ORPHA(?:NET)?\s*[:_]\s*(\d+)$", re.IGNORECASE),
]

OMIM_PATTERNS = [
    re.compile(r"^OMIM:(\d+)$", re.IGNORECASE),
    re.compile(r"^MIM:(\d+)$", re.IGNORECASE),
    re.compile(r"^OMIM\s*[:_]\s*(\d+)$", re.IGNORECASE),
]

MONDO_PATTERNS = [
    re.compile(r"^MONDO:(\d+)$", re.IGNORECASE),
    re.compile(r"^MONDO\s*[:_]\s*(\d+)$", re.IGNORECASE),
]


def normalize_xref(xref: str) -> str:
    """Normalize an ontology xref string for consistent pattern matching."""
    return xref.strip().replace("Orphanet:", "ORPHA:")


def extract_orpha_from_xrefs(xrefs: list[str]) -> str | None:
    """Extract ORPHA ID from a list of xrefs if present."""
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in ORPHA_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                return f"ORPHA:{match.group(1)}"
    return None


def extract_omim_from_xrefs(xrefs: list[str]) -> list[str]:
    """Extract OMIM IDs from a list of xrefs if present."""
    matches = []
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in OMIM_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                matches.append(f"OMIM:{match.group(1)}")
                break
    return matches


def extract_mondo_from_xrefs(xrefs: list[str]) -> list[str]:
    """Extract MONDO IDs from a list of xrefs if present."""
    matches = []
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in MONDO_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                matches.append(f"MONDO:{match.group(1)}")
                break
    return matches


def extract_orpha_from_exact_matches(exact_matches: list[str]) -> str | None:
    """Extract ORPHA ID from a list of exact matches if present."""
    for value in exact_matches:
        cleaned = value.strip()

        if "Orphanet_" in cleaned:
            number = cleaned.split("Orphanet_")[-1]
            if number.isdigit():
                return f"ORPHA:{number}"

        normalized = normalize_xref(cleaned)
        for pattern in ORPHA_PATTERNS:
            match = pattern.match(normalized)
            if match:
                return f"ORPHA:{match.group(1)}"

    return None


def build_orpha_mapping_index(
    ordo_metadata: dict[str, dict],
    mondo_metadata: dict[str, dict],
    hoom_metadata: dict[str, dict],
) -> dict[str, str]:
    """
    Build mapping index to canonical ORPHA IDs.

    Strong rule:
        - MONDO entries are mapped to ORPHA only if they have a skos:exactMatch
          to an Orphanet/ORDO ID.

    This avoids copying wrong MONDO descriptions into ORPHA profiles based only
    on weak hasDbXref mappings.

    Returns examples like:
      {
        "OMIM:301310": "ORPHA:123",
        "MONDO:0001234": "ORPHA:123"
      }
    """
    mapping: dict[str, str] = {}

    def process_metadata_entry(raw_id: str, meta: dict, source: str) -> None:
        source = source.upper()

        xrefs = meta.get("xrefs", [])
        exact_matches = meta.get("exact_matches", [])

        exact_orpha_id = extract_orpha_from_exact_matches(exact_matches)
        xref_orpha_id = extract_orpha_from_xrefs(xrefs)

        # ORPHA entries are already canonical anchors.
        if raw_id.startswith("ORPHA:"):
            orpha_id = raw_id

        # For MONDO, only trust exactMatch for mapping the MONDO ID itself.
        # This prevents weak xrefs from attaching wrong MONDO descriptions.
        elif source == "MONDO":
            orpha_id = exact_orpha_id

        # For ORDO/HOOM/other metadata, allow exactMatch first, then xref fallback.
        else:
            orpha_id = exact_orpha_id or xref_orpha_id

        if not orpha_id:
            return

        # Map the entry itself to ORPHA when it is a non-ORPHA alias.
        # Example: MONDO:0000437 -> ORPHA:102002
        if raw_id != orpha_id:
            mapping[raw_id] = orpha_id

        # Map OMIM aliases found in xrefs.
        for omim_id in extract_omim_from_xrefs(xrefs):
            mapping[omim_id] = orpha_id

        # Map MONDO aliases from non-MONDO metadata, especially ORDO metadata.
        # For MONDO metadata itself, raw_id was already handled above using exactMatch.
        if source != "MONDO":
            for mondo_id in extract_mondo_from_xrefs(xrefs):
                mapping[mondo_id] = orpha_id

    for local_id, meta in ordo_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta, source="ORDO")

    for local_id, meta in mondo_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta, source="MONDO")

    for local_id, meta in hoom_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta, source="HOOM")

    return mapping


def resolve_to_orpha(
    disease_id: str,
    mapping_index: dict[str, str],
    source_metadata: dict[str, object] | None = None,
) -> str:
    """
    Resolve to ORPHA using:
    1. already ORPHA
    2. explicit mapping index from xrefs
    3. source metadata xrefs
    4. fallback to original ID
    """
    if disease_id.startswith("ORPHA:"):
        return disease_id

    if disease_id in mapping_index:
        return mapping_index[disease_id]

    if source_metadata:
        raw_xrefs = source_metadata.get("xrefs", [])

        xrefs: list[str] = (
            [str(xref) for xref in raw_xrefs]
            if isinstance(raw_xrefs, list)
            else []
        )

        mapped_orpha = extract_orpha_from_xrefs(xrefs)

        if mapped_orpha:
            return mapped_orpha

    return disease_id


def choose_preferred_label(
    existing_label: str | None,
    incoming_label: str | None,
) -> str:
    """
    Choose the preferred label, always returning a string.
    Prefer existing non-empty label, then incoming non-empty label, then empty string.
    """
    if existing_label and existing_label.strip():
        return existing_label.strip()

    if incoming_label and incoming_label.strip():
        return incoming_label.strip()

    return ""


def merge_source_ids(
    existing: dict[str, str],
    new_source_name: str,
    new_source_id: object | None,
) -> dict[str, str]:
    """Merge source ID mappings, adding a new source and its ID to the existing mapping."""
    merged = dict(existing)

    if new_source_id is not None:
        merged[new_source_name] = str(new_source_id)

    return merged
