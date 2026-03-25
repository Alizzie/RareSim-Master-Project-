import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
SEMANTIC_DIR = PROJECT_ROOT / "outputs" / "semantic"
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_patient_terms(patient: dict, use_propagated: bool = True) -> Set[str]:
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    return set(patient.get(key, []))


def get_disease_terms(profile: dict, use_propagated: bool = True) -> Set[str]:
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    return set(profile.get(key, []))


def mica_resnik(
    term_a: str,
    term_b: str,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, Optional[str]]:
    """
    Compute Resnik similarity using the Most Informative Common Ancestor (MICA).
    Returns:
        (similarity_score, mica_term)
    """
    ancestors_a = set(ancestors.get(term_a, [])) | {term_a}
    ancestors_b = set(ancestors.get(term_b, [])) | {term_b}

    common = ancestors_a & ancestors_b
    if not common:
        return 0.0, None

    mica_term = max(common, key=lambda t: ic_values.get(t, 0.0))
    return ic_values.get(mica_term, 0.0), mica_term


def best_match_scores(
    source_terms: Set[str],
    target_terms: Set[str],
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, List[dict]]:
    """
    For each source term, find the best matching target term using Resnik.
    Returns:
        average_best_score, match_details
    """
    if not source_terms or not target_terms:
        return 0.0, []

    match_details = []

    for source_term in source_terms:
        best_score = 0.0
        best_target = None
        best_mica = None

        for target_term in target_terms:
            score, mica = mica_resnik(
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
                "score": best_score,
            }
        )

    average_score = sum(x["score"] for x in match_details) / len(match_details)
    return average_score, match_details


def resnik_bma_similarity(
    patient_terms: Set[str],
    disease_terms: Set[str],
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
) -> Tuple[float, dict]:
    """
    Compute Best Match Average (BMA):
      0.5 * [avg best(patient -> disease) + avg best(disease -> patient)]
    """
    if not patient_terms or not disease_terms:
        return 0.0, {
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
    )
    d2p_avg, d2p_matches = best_match_scores(
        disease_terms,
        patient_terms,
        ancestors,
        ic_values,
    )

    final_score = 0.5 * (p2d_avg + d2p_avg)

    explanation = {
        "patient_to_disease_avg": p2d_avg,
        "disease_to_patient_avg": d2p_avg,
        "patient_to_disease_matches": p2d_matches,
        "disease_to_patient_matches": d2p_matches,
    }
    return final_score, explanation


def summarize_explanation(explanation: dict, top_n: int = 5) -> dict:
    """
    Keep only the strongest term-level matches for compact output.
    """
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
        "patient_to_disease_avg": explanation["patient_to_disease_avg"],
        "disease_to_patient_avg": explanation["disease_to_patient_avg"],
        "top_patient_to_disease_matches": p2d,
        "top_disease_to_patient_matches": d2p,
    }


def rank_diseases(
    disease_profiles: Dict[str, dict],
    patient: dict,
    ancestors: Dict[str, List[str]],
    ic_values: Dict[str, float],
    use_propagated_terms: bool = True,
    top_k: int = 10,
) -> List[dict]:
    patient_terms = get_patient_terms(patient, use_propagated=use_propagated_terms)
    results = []

    for disease_id, profile in disease_profiles.items():
        disease_terms = get_disease_terms(
            profile,
            use_propagated=use_propagated_terms,
        )

        if not disease_terms:
            continue

        score, explanation = resnik_bma_similarity(
            patient_terms,
            disease_terms,
            ancestors,
            ic_values,
        )

        results.append(
            {
                "disease_id": disease_id,
                "label": profile.get("label"),
                "method_name": "semantic_resnik_bma",
                "score": score,
                "explanation": summarize_explanation(explanation),
                "metadata": {
                    "n_patient_terms": len(patient_terms),
                    "n_disease_terms": len(disease_terms),
                    "used_propagated_terms": use_propagated_terms,
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results[:top_k]


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main() -> None:
    disease_profiles = load_json(SHARED_DIR / "disease_profiles.json")
    ic_values = load_json(SHARED_DIR / "information_content.json")
    ancestors = load_json(SHARED_DIR / "hpo_ancestors.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    top_results = rank_diseases(
        disease_profiles=disease_profiles,
        patient=patient,
        ancestors=ancestors,
        ic_values=ic_values,
        use_propagated_terms=True,
        top_k=10,
    )

    output_path = SEMANTIC_DIR / "semantic_top10.json"
    save_json(top_results, output_path)

    print("Top semantic results:")
    for row in top_results[:10]:
        print(
            f"rank={row['rank']:>2} | "
            f"{row['disease_id']:<15} | "
            f"score={row['score']:.4f} | "
            f"{row['label']}"
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
    