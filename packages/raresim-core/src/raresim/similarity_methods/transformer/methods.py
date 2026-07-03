"""
Transformer methods — text construction and embedding backends.
DiseaseRetriever (retriever.py) uses these to build and query embeddings.

Supports:
- HuggingFace encoder models with mean pooling
- SentenceTransformer models
- AutoTokenizer for HuggingFace encoder models
"""

import hashlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from raresim.similarity_methods.transformer.config import (
    BATCH_SIZE,
    DESCRIPTION_CHAR_BUDGET,
    MAX_LENGTH,
    SENTENCE_TRANSFORMER_MODELS,
)
from raresim.types.schemas import PatientProfile
from raresim.utils.hpo_utils import get_hpo_label


# ── Text construction ────────────────────────────────────────────────────────


def select_phenotype_labels(
    terms: list[str],
    hpo_labels: dict[str, str],
) -> list[str]:
    """
    Return direct phenotype labels in stable order.

    Sorting matters because terms can come from sets, and Python's set
    iteration order is not stable across process runs.
    """
    return [get_hpo_label(term, hpo_labels) for term in sorted(terms)]


def build_patient_text(
    patient: PatientProfile,
    hpo_labels: dict[str, str],
    *,
    description_char_budget: int = DESCRIPTION_CHAR_BUDGET,
) -> str:
    """
    Build the patient text used for embedding.

    Combines:
    - direct HPO phenotype labels
    - raw clinical description, if available

    Phenotype labels are placed first so any tokenizer truncation is more
    likely to affect prose than the primary HPO signal.
    """
    raw_text = (patient.raw_text or "").strip()
    terms = list(patient.get_terms(use_propagated=False))
    phenotype_labels = select_phenotype_labels(terms, hpo_labels)

    parts = []
    if phenotype_labels:
        parts.append(f"Patient phenotypes: {'; '.join(phenotype_labels)}")
    if raw_text:
        parts.append(f"Patient description: {raw_text[:description_char_budget]}")

    return " ".join(parts).strip()


def get_disease_terms(profile: dict) -> list[str]:
    """Return direct disease HPO terms."""
    return profile.get("hpo_terms", [])


def build_disease_text(
    profile: dict,
    hpo_labels: dict[str, str],
    *,
    description_char_budget: int = DESCRIPTION_CHAR_BUDGET,
) -> str:
    """
    Build the disease text used for embedding.

    Combines:
    - disease label
    - direct HPO phenotype labels
    - merged disease description, if available
    """
    label = (profile.get("label") or "").strip()
    desc = (profile.get("merged_description") or "").strip()
    phenotype_labels = select_phenotype_labels(get_disease_terms(profile), hpo_labels)

    parts = []
    if label:
        parts.append(f"Disease: {label}")
    if phenotype_labels:
        parts.append(f"Phenotypes: {'; '.join(phenotype_labels)}")
    if desc:
        parts.append(f"Description: {desc[:description_char_budget]}")

    return " ".join(parts).strip()


def build_disease_texts(
    disease_profiles: dict[str, dict],
    hpo_labels: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Build aligned lists of disease IDs, labels, and embedding texts."""
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


# ── Embedding backends ────────────────────────────────────────────────────────


def get_device() -> str:
    """Return CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each embedding vector to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def get_model_type(model_name: str) -> str:
    """
    Route model names to the correct embedding backend.

    sentence_transformer → SentenceTransformer library
    hf_encoder           → HuggingFace AutoModel with mean pooling
    """
    if model_name in SENTENCE_TRANSFORMER_MODELS:
        return "sentence_transformer"
    return "hf_encoder"


def hash_text(text: str) -> str:
    """Create a stable hash for patient embedding cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hf_model_and_tokenizer(model_name: str):
    """Load a HuggingFace encoder model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(get_device())
    model.eval()

    return tokenizer, model


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts_hf(
    tokenizer,
    model,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Embed texts with a HuggingFace encoder and mean pooling."""
    if not texts:
        raise ValueError("No texts provided for HF embedding.")

    device = get_device()
    embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), batch_size):
            batch = texts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

    matrix = np.vstack(embeddings).astype(np.float32)
    return l2_normalize(matrix)


def load_sentence_transformer_model(model_name: str):
    """Load a SentenceTransformer model."""
    model = SentenceTransformer(model_name, device=get_device())
    model.max_seq_length = MAX_LENGTH
    return model


def embed_texts_sentence_transformer(
    model,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed texts with a SentenceTransformer model."""
    if not texts:
        raise ValueError("No texts provided for SentenceTransformer embedding.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def load_embedding_backend(model_name: str) -> dict:
    """Load the correct embedding backend for a model."""
    model_type = get_model_type(model_name)

    if model_type == "hf_encoder":
        print(f"  Loading HF encoder: {model_name}")
        tokenizer, model = load_hf_model_and_tokenizer(model_name)
        return {"model_type": model_type, "tokenizer": tokenizer, "model": model}

    if model_type == "sentence_transformer":
        print(f"  Loading SentenceTransformer: {model_name}")
        model = load_sentence_transformer_model(model_name)
        return {"model_type": model_type, "model": model}

    raise ValueError(f"Unsupported model type: {model_name}")


def embed_texts(backend: dict, texts: list[str]) -> np.ndarray:
    """Dispatch embedding calls to the appropriate backend."""
    model_type = backend["model_type"]

    if model_type == "hf_encoder":
        return embed_texts_hf(
            tokenizer=backend["tokenizer"],
            model=backend["model"],
            texts=texts,
        )

    if model_type == "sentence_transformer":
        return embed_texts_sentence_transformer(
            model=backend["model"],
            texts=texts,
        )

    raise ValueError(f"Unsupported backend type: {model_type}")
