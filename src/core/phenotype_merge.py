from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
"""Module to merge and deduplicate disease -> HPO phenotype annotations from multiple sources."""
# higher level higher priority
SOURCE_PRIORITY = {
    "HPOA": 4,
    "ORPHADATA_PRODUCT4": 3,
    "HOOM": 2,
    "MONARCH": 1,
}

FREQUENCY_RANK = {
    None: 0,
    "VERY_RARE": 1,
    "OCCASIONAL": 2,
    "FREQUENT": 3,
    "VERY_FREQUENT": 4,
    "OBLIGATE": 5,
    "EXCLUDED": -1,
}


def normalize_frequency(freq: Optional[str]) -> Optional[str]:
    if freq is None:
        return None

    freq = freq.strip()
    if not freq:
        return None

    upper = freq.upper()

    # HOOM compact / class labels
    hoom_map = {
        "VR": "VERY_RARE",
        "VERYRARE": "VERY_RARE",
        "VERY_RARE": "VERY_RARE",
        "OC": "OCCASIONAL",
        "OCCASIONAL": "OCCASIONAL",
        "F": "FREQUENT",
        "FREQUENT": "FREQUENT",
        "VF": "VERY_FREQUENT",
        "VERYFREQUENT": "VERY_FREQUENT",
        "VERY_FREQUENT": "VERY_FREQUENT",
        "OB": "OBLIGATE",
        "OBLIGATE": "OBLIGATE",
        "EXCLUDED": "EXCLUDED",
        "0%": "EXCLUDED",
        "NONE": "EXCLUDED",
        "ABSENT": "EXCLUDED",
        "NOT": "EXCLUDED",
    }
    if upper in hoom_map:
        return hoom_map[upper]

    # Product 4 textual labels
    lower = freq.lower()
    if "excluded" in lower:
        return "EXCLUDED"
    if "obligate" in lower:
        return "OBLIGATE"
    if "very frequent" in lower:
        return "VERY_FREQUENT"
    if "frequent" in lower:
        return "FREQUENT"
    if "occasional" in lower:
        return "OCCASIONAL"
    if "very rare" in lower:
        return "VERY_RARE"

    # HPOA sometimes uses ratio strings like 1/2 or HPO frequency IDs
    return freq


def is_negative_record(record: dict) -> bool:
    qualifier = (record.get("qualifier") or "").strip().upper()
    if qualifier == "NOT":
        return True

    freq = normalize_frequency(record.get("frequency_code"))
    return freq == "EXCLUDED"


def choose_best_record(records: List[dict]) -> dict:
    def sort_key(record: dict) -> Tuple[int, int]:
        source = (record.get("source") or "").upper()
        freq = normalize_frequency(record.get("frequency_code"))
        return (
            SOURCE_PRIORITY.get(source, 0),
            FREQUENCY_RANK.get(freq, 0),
        )

    return max(records, key=sort_key)


def merge_phenotype_annotation_records(
    phenotype_annotation_records: List[dict],
) -> Tuple[List[dict], Dict[str, dict], Dict[str, Set[str]]]:
    grouped = defaultdict(list)

    for record in phenotype_annotation_records:
        disease_id = record.get("database_id")
        hpo_id = record.get("hpo_id")
        if not disease_id or not hpo_id:
            continue
        grouped[(disease_id, hpo_id)].append(record)

    merged_records: List[dict] = []
    provenance_by_disease: Dict[str, dict] = defaultdict(dict)
    negative_terms_by_disease: Dict[str, Set[str]] = defaultdict(set)

    for (disease_id, hpo_id), records in grouped.items():
        positive_records = []
        had_negative_assertion = False

        normalized_all_sources = sorted(
            {
                (r.get("source") or "").upper()
                for r in records
                if r.get("source")
            }
        )

        normalized_all_frequencies = sorted(
            {
                str(normalize_frequency(r.get("frequency_code")))
                for r in records
                if r.get("frequency_code") is not None
            }
        )

        for record in records:
            normalized = dict(record)
            normalized["source"] = (record.get("source") or "").upper()
            normalized["frequency_code"] = normalize_frequency(
                record.get("frequency_code")
            )

            if is_negative_record(normalized):
                had_negative_assertion = True
                continue

            positive_records.append(normalized)

        if not positive_records:
            negative_terms_by_disease[disease_id].add(hpo_id)
            provenance_by_disease[disease_id][hpo_id] = {
                "selected_source": None,
                "selected_frequency": None,
                "all_sources": normalized_all_sources,
                "all_frequencies": normalized_all_frequencies,
                "had_negative_assertion": had_negative_assertion,
                "excluded_from_positive_annotations": True,
            }
            continue

        selected = choose_best_record(positive_records)
        merged_records.append(selected)

        provenance_by_disease[disease_id][hpo_id] = {
            "selected_source": selected.get("source"),
            "selected_frequency": selected.get("frequency_code"),
            "all_sources": normalized_all_sources,
            "all_frequencies": normalized_all_frequencies,
            "had_negative_assertion": had_negative_assertion,
            "excluded_from_positive_annotations": False,
        }

    return merged_records, provenance_by_disease, negative_terms_by_disease
