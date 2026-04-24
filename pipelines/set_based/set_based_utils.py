import json
from pathlib import Path
from typing import Dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def create_metadata(
    hpo_vocab: set, patient_terms: set, disease_terms: set, use_propagated_terms: bool
) -> dict:
    """Creates metadata for the set based pipeline"""
    metadata = {}
    metadata["pipeline_name"] = "set_based_similarity"
    metadata["use_propagated_terms"] = use_propagated_terms
    metadata["hpo_vocab_size"] = len(hpo_vocab)
    metadata["patient_terms_size"] = len(patient_terms)
    metadata["existing_hpo_terms_size"] = len(disease_terms)
    metadata["unmatched_terms"] = list(patient_terms.difference(disease_terms))
    return metadata


def get_binary_vector(terms: set, term_to_index: dict) -> Dict[str, float]:
    """Converts a set of terms into a binary vector based on the provided term_to_index mappings"""
    vec = {}
    no_match_terms = []
    for t in terms:
        if t in term_to_index:
            vec[term_to_index[t]] = 1.0
        else:
            no_match_terms.append(t)
    return vec, no_match_terms


def sort_results_by_similarity(all_results: dict, metadata: dict) -> list:
    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["similarity_score"], reverse=True
    )

    return [
        {**result, "metadata": metadata, "rank": rank}
        for rank, (_, result) in enumerate(sorted_results, start=1)
    ]
