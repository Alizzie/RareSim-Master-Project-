from typing import Dict, List, Tuple

"""
Text construction utilities for transformer retrieval.

This module builds the textual representation used for embedding:
- patient text
- disease text
"""


def unique_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving first occurrence order."""
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def hpo_terms_to_labels(
    hpo_terms: List[str],
    hpo_labels: Dict[str, str],
) -> List[str]:
    """
    Convert HPO IDs into readable phenotype labels and remove duplicates.
    """
    labels = []
    for term in hpo_terms:
        label = hpo_labels.get(term)
        if label:
            labels.append(label.strip())
    return unique_preserve_order(labels)


def build_patient_text(patient: dict, hpo_labels: Dict[str, str]) -> str:
    """
    Build the patient text used for embedding.

    Combines:
    - raw clinical description
    - HPO phenotype labels
    """
    raw_text = (patient.get("raw_text") or "").strip()
    hpo_terms = patient.get("hpo_terms", [])
    phenotype_labels = hpo_terms_to_labels(hpo_terms, hpo_labels)

    parts = []

    if raw_text:
        parts.append(f"Patient description: {raw_text}")

    if phenotype_labels:
        parts.append(f"Patient phenotypes: {'; '.join(phenotype_labels)}")

    return " ".join(parts).strip()


def build_disease_text(profile: dict, hpo_labels: Dict[str, str]) -> str:
    """
    Build the disease text used for embedding.

    Combines:
    - disease label
    - merged description
    - HPO phenotype labels
    """
    label = (profile.get("label") or "").strip()
    desc = (profile.get("merged_description") or "").strip()
    hpo_terms = profile.get("hpo_terms", [])
    phenotype_labels = hpo_terms_to_labels(hpo_terms, hpo_labels)

    parts = []

    if label:
        parts.append(f"Disease: {label}")

    if desc:
        parts.append(f"Description: {desc}")

    if phenotype_labels:
        parts.append(f"Phenotypes: {'; '.join(phenotype_labels)}")

    return " ".join(parts).strip()


def build_disease_texts(
    disease_profiles: Dict[str, dict],
    hpo_labels: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Build aligned lists of disease IDs, labels, and embedding texts.
    """
    disease_ids = []
    disease_labels = []
    disease_texts = []

    for disease_id, profile in disease_profiles.items():
        text = build_disease_text(profile, hpo_labels)
        if not text:
            continue

        disease_ids.append(disease_id)
        disease_labels.append((profile.get("label") or "").strip())
        disease_texts.append(text)

    return disease_ids, disease_labels, disease_texts
