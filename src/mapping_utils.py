import re
from typing import Dict, List, Optional
'''Utility functions for mapping disease identifiers across different ontologies and sources.'''

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
    return xref.strip().replace("Orphanet:", "ORPHA:")


def extract_orpha_from_xrefs(xrefs: List[str]) -> Optional[str]:
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in ORPHA_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                return f"ORPHA:{match.group(1)}"
    return None


def extract_omim_from_xrefs(xrefs: List[str]) -> List[str]:
    matches = []
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in OMIM_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                matches.append(f"OMIM:{match.group(1)}")
                break
    return matches


def extract_mondo_from_xrefs(xrefs: List[str]) -> List[str]:
    matches = []
    for xref in xrefs:
        cleaned = normalize_xref(xref)
        for pattern in MONDO_PATTERNS:
            match = pattern.match(cleaned)
            if match:
                matches.append(f"MONDO:{match.group(1)}")
                break
    return matches


def build_orpha_mapping_index(
    ordo_metadata: Dict[str, dict],
    mondo_metadata: Dict[str, dict],
    hoom_metadata: Dict[str, dict],
) -> Dict[str, str]:
    """
    Build mapping index to canonical ORPHA IDs from ontology xrefs.

    Returns examples like:
      {
        "OMIM:301310": "ORPHA:123",
        "MONDO:0001234": "ORPHA:123"
      }
    """
    mapping: Dict[str, str] = {}

    def process_metadata_entry(raw_id: str, meta: dict) -> None:
        xrefs = meta.get("xrefs", [])
        orpha_id = extract_orpha_from_xrefs(xrefs)

        # If local ID is already ORPHA-like, use it as anchor
        if raw_id.startswith("ORPHA:"):
            orpha_id = raw_id

        if not orpha_id:
            return

        for omim_id in extract_omim_from_xrefs(xrefs):
            mapping[omim_id] = orpha_id

        for mondo_id in extract_mondo_from_xrefs(xrefs):
            mapping[mondo_id] = orpha_id

    for local_id, meta in ordo_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta)

    for local_id, meta in mondo_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta)

    for local_id, meta in hoom_metadata.items():
        raw_id = meta.get("normalized_id", local_id)
        process_metadata_entry(raw_id, meta)

    return mapping


def resolve_to_orpha(
    disease_id: str,
    mapping_index: Dict[str, str],
    source_metadata: Optional[dict] = None,
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
        xrefs = source_metadata.get("xrefs", [])
        mapped_orpha = extract_orpha_from_xrefs(xrefs)
        if mapped_orpha:
            return mapped_orpha

    return disease_id


def choose_preferred_label(
    existing_label: Optional[str],
    incoming_label: Optional[str],
) -> Optional[str]:
    if existing_label and existing_label.strip():
        return existing_label
    if incoming_label and incoming_label.strip():
        return incoming_label.strip()
    return existing_label


def merge_source_ids(
    existing: Dict[str, str],
    new_source_name: str,
    new_source_id: str,
) -> Dict[str, str]:
    merged = dict(existing)
    merged[new_source_name] = new_source_id
    return merged
