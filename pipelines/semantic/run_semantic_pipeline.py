import json
from pathlib import Path
from typing import Optional

from semantic_methods import (
    icto_similarity,
    jaccard_similarity,
    jiang_conrath_similarity,
    lin_similarity,
    resnik_similarity,
    simgic_similarity,
)
from semantic_utils import (
    count_profile_term_status,
    load_json,
    preprocess_ancestor_sets,
    rank_diseases_bma,
    rank_diseases_set_based,
    save_json,
    summarize_patient_terms,
    summarize_top_results_namespaces,
)

"""Main script to run the semantic similarity pipeline, integrating disease
profiles, patient data, and various similarity methods.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
SEMANTIC_DIR = PROJECT_ROOT / "outputs" / "semantic"
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

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
    ic_values = load_json(SHARED_DIR / "information_content.json")
    ancestors = load_json(SHARED_DIR / "hpo_ancestors.json")
    hpo_labels = load_json(SHARED_DIR / "hpo_labels.json")

    from src.patient_loader import load_patient
    patient = load_patient(SHARED_DIR / "example_patient.json", hpo_labels)

    ancestor_sets = preprocess_ancestor_sets(ancestors)

    patient_term_summary = summarize_patient_terms(
        patient=patient,
        ic_values=ic_values,
        use_propagated_terms=USE_PROPAGATED_TERMS,
        ic_threshold=IC_THRESHOLD,
    )

    profile_term_summary = count_profile_term_status(
        disease_profiles=disease_profiles,
        ic_values=ic_values,
        use_propagated_terms=USE_PROPAGATED_TERMS,
        ic_threshold=IC_THRESHOLD,
    )

    run_config = {
        "disease_profile_file": DISEASE_PROFILE_FILE,
        "use_canonical_profiles": USE_CANONICAL_PROFILES,
        "use_propagated_terms": USE_PROPAGATED_TERMS,
        "ic_threshold": IC_THRESHOLD,
        "top_k": TOP_K,
        "n_disease_profiles_loaded": len(disease_profiles),
        "patient_id": patient.get("patient_id"),
    }

    methods_bma = {
        "semantic_resnik_bma": resnik_similarity,
        "semantic_lin_bma": lin_similarity,
        "semantic_jiang_conrath_bma": jiang_conrath_similarity,
    }

    methods_set = {
        "semantic_simgic": simgic_similarity,
        "semantic_icto": icto_similarity,
        "semantic_jaccard": jaccard_similarity,
    }

    all_results = {}
    all_diagnostics = {}

    for method_name, similarity_fn in methods_bma.items():
        results, diagnostics = rank_diseases_bma(
            disease_profiles=disease_profiles,
            patient=patient,
            ancestor_sets=ancestor_sets,
            ic_values=ic_values,
            similarity_fn=similarity_fn,
            method_name=method_name,
            use_propagated_terms=USE_PROPAGATED_TERMS,
            ic_threshold=IC_THRESHOLD,
            top_k=TOP_K,
        )
        all_results[method_name] = results
        all_diagnostics[method_name] = diagnostics
        save_json(results, SEMANTIC_DIR / f"{method_name}_top{TOP_K}.json")

    for method_name, similarity_fn in methods_set.items():
        results, diagnostics = rank_diseases_set_based(
            disease_profiles=disease_profiles,
            patient=patient,
            ic_values=ic_values,
            set_similarity_fn=similarity_fn,
            method_name=method_name,
            use_propagated_terms=USE_PROPAGATED_TERMS,
            ic_threshold=IC_THRESHOLD,
            top_k=TOP_K,
        )
        all_results[method_name] = results
        all_diagnostics[method_name] = diagnostics
        save_json(results, SEMANTIC_DIR / f"{method_name}_top{TOP_K}.json")

    top_result_namespace_summary = summarize_top_results_namespaces(all_results)

    save_json(run_config, SEMANTIC_DIR / "run_config.json")
    save_json(all_diagnostics, SEMANTIC_DIR / "run_diagnostics.json")
    save_json(all_results, SEMANTIC_DIR / f"semantic_all_methods_top{TOP_K}.json")
    save_json(patient_term_summary, SEMANTIC_DIR / "patient_term_summary.json")
    save_json(profile_term_summary, SEMANTIC_DIR / "profile_term_summary.json")
    save_json(
        top_result_namespace_summary,
        SEMANTIC_DIR / "top_result_namespace_summary.json",
    )

    print("\nRun configuration:")
    print(json.dumps(run_config, indent=2, ensure_ascii=False))

    print("\nPatient term summary:")
    print(json.dumps(patient_term_summary, indent=2, ensure_ascii=False))

    print("\nDisease profile term summary:")
    print(json.dumps(profile_term_summary, indent=2, ensure_ascii=False))

    print("\nDiagnostics:")
    for method_name, diagnostics in all_diagnostics.items():
        print(f"- {method_name}: {diagnostics}")

    print("\nTop-result namespace summary:")
    print(json.dumps(top_result_namespace_summary, indent=2, ensure_ascii=False))

    for method_name, results in all_results.items():
        print(f"\nTop results for {method_name}:")
        for row in results:
            print(
                f"rank={row['rank']:>2} | "
                f"{row['disease_id']:<15} | "
                f"score={row['score']:.4f} | "
                f"{row['label']}"
            )

    print(f"\nSaved outputs to: {SEMANTIC_DIR}")


if __name__ == "__main__":
    main()
    