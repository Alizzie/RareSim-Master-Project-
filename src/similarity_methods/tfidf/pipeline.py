"""
Pipeline:
  1. Build corpus of disease profiles (build_shared_artifacts.py)
  2. Compute TF-IDF vectors for each disease
  3. Treat the patient's HPO terms as a query document and compute their TF-IDF vector
  4. Compare patient vector to each disease vector using cosine similarity
  5. Rank diseases by similarity score
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
TFIDF_DIR = PROJECT_ROOT / "outputs" / "tfidf"
TFIDF_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# IDF (Inverse Document Frequency)
# Measures how rare or common a term is across the entire corpus
# Formula: IDF(t) = log(Total documents / Number of documents containing t)
#
# Nutshell:
#   - A term that appears in every disease profile carries no discriminative
#     signal -> IDF = log(1) = 0 -> contributes nothing to the score
#   - A term that appears in only 1 disease out of 10000 is highly specific
#     -> gets a high IDF score
#
# Our case: documents = disease profiles, terms = HPO phenotype terms
# We use propagated terms (true-path rule applied) so that ancestor terms
# are included, which naturally lowers their IDF (aka they appear in more diseases)

def compute_idf(
    disease_profiles: Dict[str, dict],
    use_propagated: bool = True,
) -> Dict[str, float]:
    """
    IDF(term) = log(total_diseases / number_of_diseases_containing_term)
    Binary TF means IDF alone carries the weighting signal.
    """
    key = "propagated_hpo_terms" if use_propagated else "hpo_terms"

    # Total
    total = len(disease_profiles)

    # Count in how many disease profiles each HPO term appears
    doc_freq: Dict[str, int] = {}
    for profile in disease_profiles.values():
        for term in set(profile.get(key, [])):  # set() avoids double-counting within one disease
            doc_freq[term] = doc_freq.get(term, 0) + 1

    # IDF: rare terms (low doc_freq) get high scores, common terms get low scores
    return {
        term: math.log(total / freq)
        for term, freq in doc_freq.items()
    }


# TF (Term Frequency)
# Measures how often a term appears in a specific document
# There are multiple ways to calculate TF: raw count, binary, log normalization,
# augmented frequency, relative frequency, but I used binary here:
#   TF(t, d) = 1 if term t is present in document d, else 0
#
# Since disease profiles are sets of HPO terms, binary TF
# is the most natural choice, a term is either assosciated with a disease or not
#
# TF-IDF(t, d) = TF(t, d) x IDF(t)
# With binary TF, this simplifies to: IDF(t) if present, else 0
# So the vector weight for a term is just its IDF value

def disease_tfidf_vector(
    terms: Set[str],
    idf: Dict[str, float],
) -> Dict[str, float]:
    """
    Build a sparse TF-IDF vector for a set of HPO terms.
    Binary TF (1 if present) × IDF = just the IDF value for each present term.
    Terms not seen during IDF computation (e.g. unseen in any disease) are ignored.
    """
    return {term: idf[term] for term in terms if term in idf}


# Cosine Similarity
# Compare the patient vector against each disease vector
# Formula: cos(A, B) = (A * B) / (||A|| x ||B||)

def cosine_similarity(
    vec_a: Dict[str, float],
    vec_b: Dict[str, float],
) -> float:
    """
    Cosine similarity between two sparse TF-IDF vectors.
    Returns a value in [0, 1] where 1 = identical term profile.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(vec_a[t] * vec_b[t] for t in vec_a if t in vec_b)

    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# Ranking
# For a new patient:
#  1. Treat the patient's HPO terms as a query document
#  2. Compute the patient's TF-IDF vector 
#  3. Compare the patient vector to every disease vector using cosine similarity
#  4. Rank diseases by descending similarity score

def rank_diseases(
    disease_profiles: Dict[str, dict],
    patient: dict,
    idf: Dict[str, float],
    use_propagated: bool = True,
    top_k: int = 10,
) -> List[dict]:
    patient_key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
    patient_terms = set(patient.get(patient_key, []))
    patient_vec = disease_tfidf_vector(patient_terms, idf)

    results = []

    for disease_id, profile in disease_profiles.items():
        disease_key = "propagated_hpo_terms" if use_propagated else "hpo_terms"
        disease_terms = set(profile.get(disease_key, []))

        if not disease_terms:
            continue

        disease_vec = disease_tfidf_vector(disease_terms, idf)

        score = cosine_similarity(patient_vec, disease_vec)

        matching_terms = [t for t in patient_terms if t in disease_terms]

        results.append({
            "disease_id": disease_id,
            "label": profile.get("label"),
            "method_name": "tfidf_cosine",
            "score": score,
            "explanation": {
                "matching_terms": matching_terms,
                "n_matching": len(matching_terms),
                "top_weighted_matches": sorted(
                    [
                        {"term": t, "tfidf_weight": idf.get(t, 0.0)}
                        for t in matching_terms
                    ],
                    key=lambda x: x["tfidf_weight"],
                    reverse=True,
                )[:5],
            },
            "metadata": {
                "n_patient_terms": len(patient_terms),
                "n_disease_terms": len(disease_terms),
                "used_propagated_terms": use_propagated,
            },
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results[:top_k]

def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main() -> None:
    print("Loading shared artifacts....")
    disease_profiles = load_json(SHARED_DIR / "disease_profiles.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    print("Computing IDF from disease profiles....")
    idf = compute_idf(disease_profiles, use_propagated=True)
    print(f"  Vocabulary size: {len(idf)} HPO terms")

    print("Ranking diseases for example patient....")
    top_results = rank_diseases(
        disease_profiles=disease_profiles,
        patient=patient,
        idf=idf,
        use_propagated=True,
        top_k=10,
    )

    output_path = TFIDF_DIR / "tfidf_top10.json"
    save_json(top_results, output_path)

    print("\nTop TF-IDF results:")
    for row in top_results:
        print(
            f"rank={row['rank']:>2} | "
            f"{row['disease_id']:<15} | "
            f"score={row['score']:.4f} | "
            f"{row['label']}"
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
