import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
TRANSFORMER_DIR = PROJECT_ROOT / "outputs" / "transformer"
TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# If True: use patient raw_text if available.
# If False: build text from HPO labels.
USE_PATIENT_RAW_TEXT_IF_AVAILABLE = True

# How many diseases to return
TOP_K = 10


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
# Text construction
# ----------------------------
def hpo_terms_to_text(hpo_terms: List[str], hpo_labels: Dict[str, str]) -> str:
    """
    Convert HPO IDs to a readable phenotype text string.
    Example:
      ["HP:0001263", "HP:0002470"] ->
      "Global developmental delay. Cerebellar ataxia."
    """
    labels = []
    for term in hpo_terms:
        label = hpo_labels.get(term)
        if label:
            labels.append(label)

    # remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)

    return ". ".join(unique_labels)


def build_patient_text(
    patient: dict,
    hpo_labels: Dict[str, str],
    use_raw_text_if_available: bool = True,
    use_propagated_terms: bool = False,
) -> str:
    """
    Build patient text either from:
    - raw clinical text
    - HPO labels
    """
    raw_text = patient.get("raw_text")
    if use_raw_text_if_available and raw_text and raw_text.strip():
        return raw_text.strip()

    key = "propagated_hpo_terms" if use_propagated_terms else "hpo_terms"
    hpo_terms = patient.get(key, [])
    return hpo_terms_to_text(hpo_terms, hpo_labels)


def build_disease_text(
    disease_profile: dict,
    hpo_labels: Dict[str, str],
    use_propagated_terms: bool = False,
) -> str:
    """
    Build disease text from:
    1. merged description if available
    2. phenotype labels from HPO terms
    3. label fallback

    We combine description + phenotype text because transformer methods
    often work better with richer textual context.
    """
    parts = []

    label = disease_profile.get("label")
    merged_description = disease_profile.get("merged_description")

    if label and label.strip():
        parts.append(label.strip())

    if merged_description and merged_description.strip():
        parts.append(merged_description.strip())

    key = "propagated_hpo_terms" if use_propagated_terms else "hpo_terms"
    hpo_terms = disease_profile.get(key, [])
    phenotype_text = hpo_terms_to_text(hpo_terms, hpo_labels)

    if phenotype_text.strip():
        parts.append(phenotype_text.strip())

    return " ".join(parts).strip()


# ----------------------------
# Embeddings + similarity
# ----------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into dense embeddings.
    """
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )


# ----------------------------
# Ranking
# ----------------------------
def rank_diseases_with_transformer(
    disease_profiles: Dict[str, dict],
    patient: dict,
    hpo_labels: Dict[str, str],
    model_name: str = MODEL_NAME,
    use_patient_raw_text_if_available: bool = True,
    use_propagated_terms_for_text: bool = False,
    top_k: int = 10,
) -> List[dict]:
    """
    Main ranking function:
    1. Build patient text
    2. Build disease texts
    3. Compute embeddings
    4. Compute cosine similarity
    5. Rank diseases
    """
    model = SentenceTransformer(model_name)

    patient_text = build_patient_text(
        patient=patient,
        hpo_labels=hpo_labels,
        use_raw_text_if_available=use_patient_raw_text_if_available,
        use_propagated_terms=use_propagated_terms_for_text,
    )

    if not patient_text.strip():
        raise ValueError("Patient text is empty. Cannot compute transformer similarity.")

    disease_ids = []
    disease_labels = []
    disease_texts = []

    for disease_id, profile in disease_profiles.items():
        disease_text = build_disease_text(
            disease_profile=profile,
            hpo_labels=hpo_labels,
            use_propagated_terms=use_propagated_terms_for_text,
        )

        if not disease_text.strip():
            continue

        disease_ids.append(disease_id)
        disease_labels.append(profile.get("label"))
        disease_texts.append(disease_text)

    if not disease_texts:
        raise ValueError("No disease texts available for transformer ranking.")

    patient_embedding = embed_texts(model, [patient_text])[0]
    disease_embeddings = embed_texts(model, disease_texts)

    results = []

    for disease_id, label, disease_text, disease_embedding in zip(
        disease_ids,
        disease_labels,
        disease_texts,
        disease_embeddings,
    ):
        score = cosine_similarity(patient_embedding, disease_embedding)

        results.append(
            {
                "disease_id": disease_id,
                "label": label,
                "method_name": "transformer_cosine_similarity",
                "score": score,
                "explanation": {
                    "patient_text": patient_text,
                    "disease_text_preview": disease_text[:500],
                    "similarity_type": "cosine",
                },
                "metadata": {
                    "model_name": model_name,
                    "used_patient_raw_text_if_available": use_patient_raw_text_if_available,
                    "used_propagated_terms_for_text": use_propagated_terms_for_text,
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
    hpo_labels = load_json(SHARED_DIR / "hpo_labels.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    top_results = rank_diseases_with_transformer(
        disease_profiles=disease_profiles,
        patient=patient,
        hpo_labels=hpo_labels,
        model_name=MODEL_NAME,
        use_patient_raw_text_if_available=USE_PATIENT_RAW_TEXT_IF_AVAILABLE,
        use_propagated_terms_for_text=False,
        top_k=TOP_K,
    )

    output_path = TRANSFORMER_DIR / "transformer_top10.json"
    save_json(top_results, output_path)

    print(f"Model: {MODEL_NAME}")
    print("Top transformer results:")
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
