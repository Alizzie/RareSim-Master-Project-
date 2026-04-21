"""
Main script to run the semantic similarity pipeline, integrating disease
profiles, patient data, and cosine similarity calculations to rank diseases based on HPO term overlap.
"""

from pathlib import Path
from typing import Optional

from vector_similarity_methods import cosine_similarity_dense
from set_based_utils import (
    load_json,
    save_json,
    _create_metadata,
    _get_binary_vector,
    _sort_results_by_similarity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
SETBASED_DIR = PROJECT_ROOT / "outputs" / "set_based"
SETBASED_DIR.mkdir(parents=True, exist_ok=True)

USE_CANONICAL_PROFILES = True
USE_PROPAGATED_TERMS = True
TOP_K = 10
IC_THRESHOLD: Optional[float] = 1.5

DISEASE_PROFILE_FILE = (
    "canonical_disease_profiles.json"
    if USE_CANONICAL_PROFILES
    else "disease_profiles.json"
)


def main() -> None:
    disease_profiles = load_json(SHARED_DIR / DISEASE_PROFILE_FILE)
    patient = load_json(SHARED_DIR / "example_patient.json")
    loaded_hpo_terms = load_json(SHARED_DIR / "hpo_labels.json")

    terms_key = "propagated_hpo_terms" if USE_PROPAGATED_TERMS else "hpo_terms"
    patient_terms = set(patient.get(terms_key, []))

    hpo_vocab = sorted(set(loaded_hpo_terms.keys()).union(patient_terms))
    term_to_index = {term: idx for idx, term in enumerate(hpo_vocab)}

    patient_vec, _ = _get_binary_vector(patient_terms, term_to_index)

    metadata = _create_metadata(
        hpo_vocab=hpo_vocab,
        patient_terms=patient_terms,
        disease_terms=set(loaded_hpo_terms.keys()),
        use_propagated_terms=USE_PROPAGATED_TERMS,
    )
    all_results = {}

    for disease_id, profile in disease_profiles.items():
        disease_terms = set(profile.get(terms_key, []))
        disease_vec, _ = _get_binary_vector(disease_terms, term_to_index)

        similarity_score = cosine_similarity_dense(patient_vec, disease_vec)
        matched_hpo_terms = patient_terms.intersection(disease_terms)

        all_results[disease_id] = {
            "disease_id": disease_id,
            "label": profile.get("label", ""),
            "similarity_score": similarity_score,
            "matched_terms": list(matched_hpo_terms),
        }

    results = _sort_results_by_similarity(all_results, metadata)
    top_k_summary = results[:TOP_K]

    save_json(results, SETBASED_DIR / "set_based_similarity_results.json")
    save_json(top_k_summary, SETBASED_DIR / "set_based_similarity_top_k_summary.json")


if __name__ == "__main__":
    main()
