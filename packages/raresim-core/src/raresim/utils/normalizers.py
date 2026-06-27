"""
Functions to normalize and standardize identifiers for diseases and HPO terms,
ensuring consistency across different data sources.
"""

import re
from typing import Optional

HPO_PATTERN = re.compile(r"^HP:\d{7}$")


def normalize_hpo_id(hpo_id: str) -> Optional[str]:
    """Normalize an HPO ID to the standard format (e.g. HP:0004322). Returns None if the ID cannot be normalized."""
    if not hpo_id:
        return None

    hpo_id = hpo_id.strip().replace("_", ":").upper()

    if hpo_id.startswith("HTTP") and "HP_" in hpo_id:
        hpo_id = hpo_id.split("/")[-1].replace("_", ":")

    if HPO_PATTERN.match(hpo_id):
        return hpo_id

    return None


def normalize_disease_id(raw_id: object) -> str | None:
    """Normalize a disease ID to a stable display format."""
    if raw_id is None:
        return None

    value = str(raw_id).strip()
    if not value:
        return None

    if value.startswith("http"):
        value = value.rstrip("/").split("/")[-1]

    if value.startswith("Orphanet_"):
        return f"ORPHA:{value.split('_', maxsplit=1)[1]}"

    if value.startswith("ORPHA_"):
        return f"ORPHA:{value.split('_', maxsplit=1)[1]}"

    if value.startswith("ORPHANET_"):
        return f"ORPHA:{value.split('_', maxsplit=1)[1]}"

    if value.startswith("MONDO_"):
        return f"MONDO:{value.split('_', maxsplit=1)[1]}"

    if value.startswith("OMIM_"):
        return f"OMIM:{value.split('_', maxsplit=1)[1]}"

    if value.startswith("Orphanet:"):
        return f"ORPHA:{value.split(':', maxsplit=1)[1]}"

    if value.startswith("ORPHANET:"):
        return f"ORPHA:{value.split(':', maxsplit=1)[1]}"

    if value.startswith("MIM:"):
        return f"OMIM:{value.split(':', maxsplit=1)[1]}"

    if value.startswith(("ORPHA:", "OMIM:", "MONDO:", "DECIPHER:", "DOID:")):
        return value

    if value.isdigit():
        return f"ORPHA:{value}"

    return value


def normalize_owl_local_id(local_id: str) -> str:
    """Normalize an OWL local ID to a standard disease ID format if possible."""
    normalized = normalize_disease_id(local_id)
    return normalized if normalized is not None else local_id

