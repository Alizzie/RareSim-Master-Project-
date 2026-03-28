import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

"""
Pipeline to compute semantic similarity between an example patient profile
and disease profiles using various methods:
- Resnik + BMA
- Lin + BMA
- Jiang-Conrath similarity + BMA
- simGIC
- ICTO
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
SEMANTIC_DIR = PROJECT_ROOT / "outputs" / "semantic"
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

IC_THRESHOLD = 1.5  # tune between 1.0 and 2.0


# ----------------------------
# Basic I/O
# ----------------------------
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


# ----------------------------
# Profile helpers
# ----------------------------
def get_patient_terms(
    patient: dict,
    ic_values: Dict[str, float],
    use_propagated: bool = True,
) -> Set[str]:
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    terms = set(patient.get(key, []))
    return {term for term in terms if ic_values.get(term, 0.0) >= IC_THRESHOLD}


def get_disease_terms(
    profile: dict,
    ic_values: Dict[str, float],
    use_propagated: bool = True,
) -> Set[str]:
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    terms = set(profile.get(key, []))
    return {term for term in terms if ic_values.get(term, 0.0) >= IC_THRESHOLD}


def get_ancestors_inclusive(
    term: str,
    ancestors: Dict[str, List[str]],
) -> Set[str]:
    return set(ancestors.get(term, [])) | {term}


# ----------------------------
# MICA / shared ancestor logic
# ----------------------------
def get_common_ancestors(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
) -> Set[str]:
    ancestors_a = get_ancestors_inclusive(term_a, ancestors)
    ancestors_b = get_ancestors_inclusive(term_b, ancestors)
    return ancestors_a & ancestors_b


def get_mica(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[Optional[str], float]:
    common = get_common_ancestors(term_a, term_b, ancestors)
    if not common:
        return None, 0.0

    mica_term = max(common, key=lambda t: ic_values.get(t, 0.0))
    mica_ic = ic_values.get(mica_term, 0.0)
    return mica_term, mica_ic


# ----------------------------
# Term-term similarities
# ----------------------------
def resnik_similarity(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    mica_term, mica_ic = get_mica(term_a, term_b, ancestors, ic_values)
    return mica_ic, mica_term


def lin_similarity(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    mica_term, mica_ic = get_mica(term_a, term_b, ancestors, ic_values)
    if mica_term is None:
        return 0.0, None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    denom = ic_a + ic_b

    if denom == 0.0:
        return 0.0, mica_term

    score = (2.0 * mica_ic) / denom
    return score, mica_term


def jiang_conrath_distance(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Jiang-Conrath distance:
        dist = IC(a) + IC(b) - 2 * IC(MICA)
    Lower is more similar.
    """
    mica_term, mica_ic = get_mica(term_a, term_b, ancestors, ic_values)
    if mica_term is None:
        return float("inf"), None

    ic_a = ic_values.get(term_a, 0.0)
    ic_b = ic_values.get(term_b, 0.0)
    distance = ic_a + ic_b - 2.0 * mica_ic
    return max(distance, 0.0), mica_term


def jiang_conrath_similarity(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Convert Jiang-Conrath distance into similarity:
        sim = 1 / (1 + dist)
    So similarity is in (0, 1].
    """
    distance, mica_term = jiang_conrath_distance(
        term_a,
        term_b,
        ancestors,
        ic_values,
    )
    if math.isinf(distance):
        return 0.0, None

    score = 1.0 / (1.0 + distance)
    return score, mica_term


# ----------------------------
# BMA aggregation
# ----------------------------
SimilarityFn = Callable[
    [str, str, Dict[str, List[str]], Dict[str, float]],
    Tuple[float, Optional[str]],
]


def best_match_scores(
    source_terms: Set[str],
    target_terms: Set[str],
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
    similarity_fn: SimilarityFn,
) -> Tuple[float, List[dict]]:
    if not source_terms or not target_terms:
        return 0.0, []

    match_details = []

    for source_term in source_terms:
        best_score = -1.0
        best_target = None
        best_mica = None

        for target_term in target_terms:
            score, mica = similarity_fn(
                source_term,
                target_term,
                ancestors,
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
                "score": best_score if best_score >= 0.0 else 0.0,
            }
        )

    average_score = sum(x["score"] for x in match_details) / len(match_details)
    return average_score, match_details


def bma_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ancestors: Dict[str, List[str]],
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

    p2d_avg, p2d_matches = best_match_scores(
        patient_terms,
        disease_terms,
        ancestors,
        ic_values,
        similarity_fn,
    )
    d2p_avg, d2p_matches = best_match_scores(
        disease_terms,
        patient_terms,
        ancestors,
        ic_values,
        similarity_fn,
    )

    final_score = 0.5 * (p2d_avg + d2p_avg)

    explanation = {
        "method": method_name,
        "patient_to_disease_avg": p2d_avg,
        "disease_to_patient_avg": d2p_avg,
        "patient_to_disease_matches": p2d_matches,
        "disease_to_patient_matches": d2p_matches,
    }
    return final_score, explanation


# ----------------------------
# Set-based IC methods
# ----------------------------
def sum_ic(terms: Set[str], ic_values: Dict[str, float]) -> float:
    return sum(ic_values.get(term, 0.0) for term in terms)


def simgic_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: Dict[str, float],
) -> Tuple[float, dict]:
    """
    simGIC(A, B) = sum(IC(intersection)) / sum(IC(union))
    """
    if not patient_terms or not disease_terms:
        return 0.0, {
            "intersection_size": 0,
            "union_size": 0,
            "intersection_ic_sum": 0.0,
            "union_ic_sum": 0.0,
        }

    inter = patient_terms & disease_terms
    union = patient_terms | disease_terms

    inter_ic = sum_ic(inter, ic_values)
    union_ic = sum_ic(union, ic_values)

    score = 0.0 if union_ic == 0.0 else inter_ic / union_ic

    explanation = {
        "intersection_size": len(inter),
        "union_size": len(union),
        "intersection_ic_sum": inter_ic,
        "union_ic_sum": union_ic,
        "top_shared_terms": sorted(
            inter,
            key=lambda t: ic_values.get(t, 0.0),
            reverse=True,
        )[:10],
    }
    return score, explanation


def icto_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ic_values: Dict[str, float],
) -> Tuple[float, dict]:
    """
    IC-weighted overlap coefficient:
        ICTO(A, B) = sum(IC(intersection)) / min(sum(IC(A)), sum(IC(B)))
    """
    if not patient_terms or not disease_terms:
        return 0.0, {
            "intersection_size": 0,
            "patient_ic_sum": 0.0,
            "disease_ic_sum": 0.0,
            "intersection_ic_sum": 0.0,
        }

    inter = patient_terms & disease_terms
    patient_ic = sum_ic(patient_terms, ic_values)
    disease_ic = sum_ic(disease_terms, ic_values)
    inter_ic = sum_ic(inter, ic_values)

    denom = min(patient_ic, disease_ic)
    score = 0.0 if denom == 0.0 else inter_ic / denom

    explanation = {
        "intersection_size": len(inter),
        "patient_ic_sum": patient_ic,
        "disease_ic_sum": disease_ic,
        "intersection_ic_sum": inter_ic,
        "top_shared_terms": sorted(
            inter,
            key=lambda t: ic_values.get(t, 0.0),
            reverse=True,
        )[:10],
    }
    return score, explanation


# ----------------------------
# Explanation summarization
# ----------------------------
def summarize_bma_explanation(explanation: dict, top_n: int = 5) -> dict:
    p2d = sorted(
        explanation["patient_to_disease_matches"],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_n]

    d2p = sorted(
        explanation["disease_to_patient_matches"],
        key=lambda x: x["score"],
        reverse=True,
    )[:top_n]

    return {
        "method": explanation["method"],
        "patient_to_disease_avg": explanation["patient_to_disease_avg"],
        "disease_to_patient_avg": explanation["disease_to_patient_avg"],
        "top_patient_to_disease_matches": p2d,
        "top_disease_to_patient_matches": d2p,
    }


# ----------------------------
# Ranking functions
# ----------------------------
def rank_diseases_bma(
    disease_profiles: Dict[str, dict],
    patient: dict,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
    similarity_fn: SimilarityFn,
    method_name: str,
    use_propagated_terms: bool = True,
    top_k: int = 10,
) -> List[dict]:
    patient_terms = get_patient_terms(
        patient,
        ic_values,
        use_propagated=use_propagated_terms,
    )
    results = []

    if not patient_terms:
        print(
            f"Warning: no patient terms remain after IC filtering "
            f"(threshold={IC_THRESHOLD})."
        )
        return results

    for disease_id, profile in disease_profiles.items():
        disease_terms = get_disease_terms(
            profile,
            ic_values,
            use_propagated=use_propagated_terms,
        )
        if not disease_terms:
            continue

        score, explanation = bma_similarity(
            patient_terms,
            disease_terms,
            ancestors,
            ic_values,
            similarity_fn,
            method_name=method_name,
        )

        results.append(
            {
                "disease_id": disease_id,
                "label": profile.get("label"),
                "method_name": method_name,
                "score": score,
                "explanation": summarize_bma_explanation(explanation),
                "metadata": {
                    "n_patient_terms": len(patient_terms),
                    "n_disease_terms": len(disease_terms),
                    "used_propagated_terms": use_propagated_terms,
                    "ic_threshold": IC_THRESHOLD,
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results[:top_k]


def rank_diseases_set_based(
    disease_profiles: Dict[str, dict],
    patient: dict,
    ic_values: Dict[str, float],
    set_similarity_fn: Callable[[Set[str], Set[str], Dict[str, float]], Tuple[float, dict]],
    method_name: str,
    use_propagated_terms: bool = True,
    top_k: int = 10,
) -> List[dict]:
    patient_terms = get_patient_terms(
        patient,
        ic_values,
        use_propagated=use_propagated_terms,
    )
    results = []

    if not patient_terms:
        print(
            f"Warning: no patient terms remain after IC filtering "
            f"(threshold={IC_THRESHOLD})."
        )
        return results

    for disease_id, profile in disease_profiles.items():
        disease_terms = get_disease_terms(
            profile,
            ic_values,
            use_propagated=use_propagated_terms,
        )
        if not disease_terms:
            continue

        score, explanation = set_similarity_fn(
            patient_terms,
            disease_terms,
            ic_values,
        )

        results.append(
            {
                "disease_id": disease_id,
                "label": profile.get("label"),
                "method_name": method_name,
                "score": score,
                "explanation": explanation,
                "metadata": {
                    "n_patient_terms": len(patient_terms),
                    "n_disease_terms": len(disease_terms),
                    "used_propagated_terms": use_propagated_terms,
                    "ic_threshold": IC_THRESHOLD,
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results[:top_k]


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    disease_profiles = load_json(SHARED_DIR / "disease_profiles.json")
    ic_values = load_json(SHARED_DIR / "information_content.json")
    ancestors = load_json(SHARED_DIR / "hpo_ancestors.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    methods_bma = {
        "semantic_resnik_bma": resnik_similarity,
        "semantic_lin_bma": lin_similarity,
        "semantic_jiang_conrath_bma": jiang_conrath_similarity,
    }

    methods_set = {
        "semantic_simgic": simgic_similarity,
        "semantic_icto": icto_similarity,
    }

    all_results = {}

    for method_name, similarity_fn in methods_bma.items():
        results = rank_diseases_bma(
            disease_profiles=disease_profiles,
            patient=patient,
            ancestors=ancestors,
            ic_values=ic_values,
            similarity_fn=similarity_fn,
            method_name=method_name,
            use_propagated_terms=True,
            top_k=10,
        )
        all_results[method_name] = results
        save_json(results, SEMANTIC_DIR / f"{method_name}_top10.json")

    for method_name, similarity_fn in methods_set.items():
        results = rank_diseases_set_based(
            disease_profiles=disease_profiles,
            patient=patient,
            ic_values=ic_values,
            set_similarity_fn=similarity_fn,
            method_name=method_name,
            use_propagated_terms=True,
            top_k=10,
        )
        all_results[method_name] = results
        save_json(results, SEMANTIC_DIR / f"{method_name}_top10.json")

    print(f"Using IC threshold: {IC_THRESHOLD}")

    for method_name, results in all_results.items():
        print(f"\nTop results for {method_name}:")
        for row in results[:10]:
            print(
                f"rank={row['rank']:>2} | "
                f"{row['disease_id']:<15} | "
                f"score={row['score']:.4f} | "
                f"{row['label']}"
            )

    save_json(all_results, SEMANTIC_DIR / "semantic_all_methods_top10.json")
    print(f"\nSaved outputs to: {SEMANTIC_DIR}")


if __name__ == "__main__":
    main()
    