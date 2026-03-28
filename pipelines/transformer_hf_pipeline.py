import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
TRANSFORMER_DIR = PROJECT_ROOT / "outputs" / "transformer"
TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)

# different models to try:
# "dmis-lab/biobert-base-cased-v1.1"
# "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# "emilyalsentzer/Bio_ClinicalBERT"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

TOP_K = 10
MAX_LENGTH = 256


# ----------------------------
# Basic I/O
# ----------------------------
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ----------------------------
# Text construction
# ----------------------------
def hpo_terms_to_text(hpo_terms: List[str], hpo_labels: Dict[str, str]) -> str:
    labels = []
    for t in hpo_terms:
        if t in hpo_labels:
            labels.append(hpo_labels[t])

    # remove duplicates
    seen = set()
    unique = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            unique.append(l)

    return ". ".join(unique)


def build_patient_text(patient: dict, hpo_labels: Dict[str, str]) -> str:
    raw = patient.get("raw_text")
    if raw and raw.strip():
        return raw.strip()

    terms = patient.get("hpo_terms", [])
    return hpo_terms_to_text(terms, hpo_labels)


def build_disease_text(profile: dict, hpo_labels: Dict[str, str]) -> str:
    parts = []

    label = profile.get("label")
    desc = profile.get("merged_description")

    if label:
        parts.append(label)
    if desc:
        parts.append(desc)

    terms = profile.get("hpo_terms", [])
    phenotype_text = hpo_terms_to_text(terms, hpo_labels)
    if phenotype_text:
        parts.append(phenotype_text)

    return " ".join(parts)


# ----------------------------
# Embedding (HF + mean pooling)
# ----------------------------
def mean_pool(last_hidden_state, attention_mask):
    """
    Mean pooling with attention mask
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    tokenizer,
    model,
    texts: List[str],
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    embeddings = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


# ----------------------------
# Similarity
# ----------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------
# Ranking
# ----------------------------
def rank_diseases_transformer_hf(
    disease_profiles: Dict[str, dict],
    patient: dict,
    hpo_labels: Dict[str, str],
) -> List[dict]:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    patient_text = build_patient_text(patient, hpo_labels)
    if not patient_text.strip():
        raise ValueError("Empty patient text")

    disease_ids = []
    disease_labels = []
    disease_texts = []

    for did, profile in disease_profiles.items():
        txt = build_disease_text(profile, hpo_labels)
        if not txt.strip():
            continue

        disease_ids.append(did)
        disease_labels.append(profile.get("label"))
        disease_texts.append(txt)

    # embeddings
    patient_emb = embed_texts(tokenizer, model, [patient_text])[0]
    disease_embs = embed_texts(tokenizer, model, disease_texts)

    results = []

    for did, label, txt, emb in zip(
        disease_ids, disease_labels, disease_texts, disease_embs
    ):
        score = cosine_similarity(patient_emb, emb)

        results.append(
            {
                "disease_id": did,
                "label": label,
                "method_name": "transformer_hf_mean_pooling",
                "score": score,
                "explanation": {
                    "patient_text": patient_text[:300],
                    "disease_text_preview": txt[:300],
                },
                "metadata": {
                    "model_name": MODEL_NAME,
                    "pooling": "mean",
                },
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results, start=1):
        r["rank"] = i

    return results[:TOP_K]


# ----------------------------
# Main
# ----------------------------
def main():
    disease_profiles = load_json(SHARED_DIR / "disease_profiles.json")
    hpo_labels = load_json(SHARED_DIR / "hpo_labels.json")
    patient = load_json(SHARED_DIR / "example_patient.json")

    results = rank_diseases_transformer_hf(
        disease_profiles,
        patient,
        hpo_labels,
    )

    out_path = TRANSFORMER_DIR / "transformer_hf_top10.json"
    save_json(results, out_path)

    print(f"\nModel: {MODEL_NAME}")
    print("Top results (HF transformer):")
    for r in results:
        print(
            f"rank={r['rank']:>2} | {r['disease_id']:<15} | "
            f"score={r['score']:.4f} | {r['label']}"
        )

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
