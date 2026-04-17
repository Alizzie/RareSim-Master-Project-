import re
from typing import Optional
'''Functions to normalize and standardize identifiers for diseases and HPO terms, ensuring consistency across different data sources.'''

HPO_PATTERN = re.compile(r"^HP:\d{7}$")


def normalize_hpo_id(hpo_id: str) -> Optional[str]:
    if not hpo_id:
        return None

    hpo_id = hpo_id.strip().replace("_", ":").upper()

    if hpo_id.startswith("HTTP") and "HP_" in hpo_id:
        hpo_id = hpo_id.split("/")[-1].replace("_", ":")

    if HPO_PATTERN.match(hpo_id):
        return hpo_id

    return None


def normalize_disease_id(raw_id: str) -> Optional[str]:
    if not raw_id:
        return None

    raw_id = raw_id.strip()

    if raw_id.startswith("Orphanet:"):
        return raw_id.replace("Orphanet:", "ORPHA:")

    if raw_id.startswith("ORPHA:"):
        return raw_id

    if raw_id.startswith("OMIM:"):
        return raw_id

    if raw_id.startswith("MONDO:"):
        return raw_id

    if raw_id.startswith("DECIPHER:"):
        return raw_id

    if raw_id.isdigit():
        return f"ORPHA:{raw_id}"

    return raw_id


def normalize_owl_local_id(local_id: str) -> str:
    if local_id.startswith("Orphanet_"):
        return "ORPHA:" + local_id.split("_", maxsplit=1)[1]

    if local_id.startswith("MONDO_"):
        return "MONDO:" + local_id.split("_", maxsplit=1)[1]

    return local_id
