"""Utility functions for semantic similarity calculations and disease profile handling:"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from collections import Counter


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def get_namespace(disease_id: str) -> str:
    if ":" in disease_id:
        return disease_id.split(":", 1)[0]
    return "UNKNOWN"


def count_profile_term_status(
    disease_profiles: Dict[str, dict],
    ic_values: Dict[str, float],
    use_propagated_terms: bool,
    ic_threshold: Optional[float],
) -> dict:
    n_total = len(disease_profiles)
    n_with_terms_before_filter = 0
    n_with_terms_after_filter = 0
    n_lost_all_terms_due_to_threshold = 0

    for profile in disease_profiles.values():
        key = "propagated_hpo_terms" if use_propagated_terms else "hpo_terms"
        original_terms = set(profile.get(key, []))
        filtered_terms = filter_terms_by_ic(original_terms, ic_values, ic_threshold)

        if original_terms:
            n_with_terms_before_filter += 1
        if filtered_terms:
            n_with_terms_after_filter += 1
        if original_terms and not filtered_terms:
            n_lost_all_terms_due_to_threshold += 1

    return {
        "n_total_profiles": n_total,
        "n_with_terms_before_filter": n_with_terms_before_filter,
        "n_with_terms_after_filter": n_with_terms_after_filter,
        "n_lost_all_terms_due_to_threshold": n_lost_all_terms_due_to_threshold,
    }


def summarize_patient_terms(
    patient: dict,
    ic_values: Dict[str, float],
    use_propagated_terms: bool,
    ic_threshold: Optional[float],
) -> dict:
    key = "propagated_hpo_terms" if use_propagated_terms else "hpo_terms"
    original_terms = set(patient.get(key, []))
    filtered_terms = filter_terms_by_ic(original_terms, ic_values, ic_threshold)

    removed_terms = sorted(original_terms - filtered_terms)
    kept_terms = sorted(filtered_terms)

    return {
        "n_terms_before_filter": len(original_terms),
        "n_terms_after_filter": len(filtered_terms),
        "kept_terms": kept_terms,
        "removed_terms": removed_terms,
    }


def count_namespaces_in_results(results: List[dict]) -> dict:
    counter = Counter(get_namespace(row["disease_id"]) for row in results)
    return dict(counter)


def summarize_top_results_namespaces(
    all_results: Dict[str, List[dict]],
) -> dict:
    summary = {}
    for method_name, results in all_results.items():
        summary[method_name] = count_namespaces_in_results(results)
    return summary


def filter_terms_by_ic(
    terms: Set[str],
    ic_values: Dict[str, float],
    ic_threshold: Optional[float],
) -> Set[str]:
    if ic_threshold is None:
        return set(terms)
    return {term for term in terms if ic_values.get(term, 0.0) >= ic_threshold}


def get_profile_terms(
    profile: dict,
    ic_values: Dict[str, float],
    use_propagated: bool,
    ic_threshold: Optional[float],
) -> Set[str]:
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    terms = set(profile.get(key, []))
    return filter_terms_by_ic(terms, ic_values, ic_threshold)


def preprocess_ancestor_sets(
    ancestors: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    return {
        term: set(parent_terms) | {term} for term, parent_terms in ancestors.items()
    }


def get_ancestors_inclusive(
    term: str,
    ancestor_sets: Dict[str, Set[str]],
) -> Set[str]:
    return ancestor_sets.get(term, {term})


def sum_ic(terms: Set[str], ic_values: Dict[str, float]) -> float:
    return sum(ic_values.get(term, 0.0) for term in terms)


def summarize_bma_explanation(explanation: dict, top_n: int = 5) -> dict:
    patient_to_disease = sorted(
        explanation["patient_to_disease_matches"],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_n]

    disease_to_patient = sorted(
        explanation["disease_to_patient_matches"],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_n]

    return {
        "method": explanation["method"],
        "patient_to_disease_avg": explanation["patient_to_disease_avg"],
        "disease_to_patient_avg": explanation["disease_to_patient_avg"],
        "top_patient_to_disease_matches": patient_to_disease,
        "top_disease_to_patient_matches": disease_to_patient,
    }


def build_result_row(
    disease_id: str,
    profile: dict,
    method_name: str,
    score: float,
    explanation: dict,
    patient_terms: Set[str],
    disease_terms: Set[str],
    use_propagated_terms: bool,
    ic_threshold: Optional[float],
) -> dict:
    return {
        "disease_id": disease_id,
        "label": profile.get("label"),
        "method_name": method_name,
        "score": score,
        "explanation": explanation,
        "metadata": {
            "n_patient_terms": len(patient_terms),
            "n_disease_terms": len(disease_terms),
            "used_propagated_terms": use_propagated_terms,
            "ic_threshold": ic_threshold,
        },
    }


SimilarityFn = Callable[
    [str, str, Dict[str, Set[str]], Dict[str, float]],
    Tuple[float, str | None],
]

SetSimilarityFn = Callable[
    [Set[str], Set[str], Dict[str, float]],
    Tuple[float, dict],
]


def best_match_scores(
    source_terms: Set[str],
    target_terms: Set[str],
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
    similarity_fn: SimilarityFn,
) -> Tuple[float, List[dict]]:
    if not source_terms or not target_terms:
        return 0.0, []

    match_details = []

    for source_term in source_terms:
        best_score = 0.0
        best_target = None
        best_mica = None

        for target_term in target_terms:
            score, mica = similarity_fn(
                source_term,
                target_term,
                ancestor_sets,
                ic_values,
            )
            if score > best_score:
                best_score = score
                best_target = target_term
                best_mica = mica

        match_details.append(
            {
                "source_term": source_term,
                "best_target_term": best_target,
                "mica_term": best_mica,
                "score": best_score,
            }
        )

    average_score = sum(x["score"] for x in match_details) / len(match_details)
    return average_score, match_details


def bma_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
    similarity_fn: SimilarityFn,
    method_name: str,
) -> Tuple[float, dict]:
    if not patient_terms or not disease_terms:
        return 0.0, {
            "method": method_name,
            "patient_to_disease_matches": [],
            "disease_to_patient_matches": [],
            "patient_to_disease_avg": 0.0,
            "disease_to_patient_avg": 0.0,
        }

    patient_to_disease_avg, patient_to_disease_matches = best_match_scores(
        patient_terms,
        disease_terms,
        ancestor_sets,
        ic_values,
        similarity_fn,
    )
    disease_to_patient_avg, disease_to_patient_matches = best_match_scores(
        disease_terms,
        patient_terms,
        ancestor_sets,
        ic_values,
        similarity_fn,
    )

    final_score = 0.5 * (patient_to_disease_avg + disease_to_patient_avg)

    explanation = {
        "method": method_name,
        "patient_to_disease_avg": patient_to_disease_avg,
        "disease_to_patient_avg": disease_to_patient_avg,
        "patient_to_disease_matches": patient_to_disease_matches,
        "disease_to_patient_matches": disease_to_patient_matches,
    }
    return final_score, explanation


def rank_diseases_bma(
    disease_profiles: Dict[str, dict],
    patient: dict,
    ancestor_sets: Dict[str, Set[str]],
    ic_values: Dict[str, float],
    similarity_fn: SimilarityFn,
    method_name: str,
    use_propagated_terms: bool,
    ic_threshold: Optional[float],
    top_k: int,
) -> Tuple[List[dict], dict]:
    patient_terms = get_profile_terms(
        patient,
        ic_values,
        use_propagated_terms,
        ic_threshold,
    )

    results = []
    skipped_no_terms = 0

    if not patient_terms:
        return [], {
            "n_patient_terms_after_filtering": 0,
            "n_diseases_considered": 0,
            "n_diseases_skipped_no_terms": 0,
        }

    for disease_id, profile in disease_profiles.items():
        disease_terms = get_profile_terms(
            profile,
            ic_values,
            use_propagated_terms,
            ic_threshold,
        )

        if not disease_terms:
            skipped_no_terms += 1
            continue

        score, explanation = bma_similarity(
            patient_terms,
            disease_terms,
            ancestor_sets,
            ic_values,
            similarity_fn,
            method_name,
        )

        results.append(
            build_result_row(
                disease_id=disease_id,
                profile=profile,
                method_name=method_name,
                score=score,
                explanation=summarize_bma_explanation(explanation),
                patient_terms=patient_terms,
                disease_terms=disease_terms,
                use_propagated_terms=use_propagated_terms,
                ic_threshold=ic_threshold,
            )
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    diagnostics = {
        "n_patient_terms_after_filtering": len(patient_terms),
        "n_diseases_considered": len(results) + skipped_no_terms,
        "n_diseases_skipped_no_terms": skipped_no_terms,
    }
    return results[:top_k], diagnostics


def rank_diseases_set_based(
    disease_profiles: Dict[str, dict],
    patient: dict,
    ic_values: Dict[str, float],
    set_similarity_fn: SetSimilarityFn,
    method_name: str,
    use_propagated_terms: bool,
    ic_threshold: Optional[float],
    top_k: int,
) -> Tuple[List[dict], dict]:
    patient_terms = get_profile_terms(
        patient,
        ic_values,
        use_propagated_terms,
        ic_threshold,
    )

    results = []
    skipped_no_terms = 0

    if not patient_terms:
        return [], {
            "n_patient_terms_after_filtering": 0,
            "n_diseases_considered": 0,
            "n_diseases_skipped_no_terms": 0,
        }

    for disease_id, profile in disease_profiles.items():
        disease_terms = get_profile_terms(
            profile,
            ic_values,
            use_propagated_terms,
            ic_threshold,
        )

        if not disease_terms:
            skipped_no_terms += 1
            continue

        score, explanation = set_similarity_fn(
            patient_terms,
            disease_terms,
            ic_values,
        )

        results.append(
            build_result_row(
                disease_id=disease_id,
                profile=profile,
                method_name=method_name,
                score=score,
                explanation=explanation,
                patient_terms=patient_terms,
                disease_terms=disease_terms,
                use_propagated_terms=use_propagated_terms,
                ic_threshold=ic_threshold,
            )
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    diagnostics = {
        "n_patient_terms_after_filtering": len(patient_terms),
        "n_diseases_considered": len(results) + skipped_no_terms,
        "n_diseases_skipped_no_terms": skipped_no_terms,
    }
    return results[:top_k], diagnostics
