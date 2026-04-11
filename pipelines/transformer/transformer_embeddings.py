import hashlib
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, BertTokenizer, BertTokenizerFast

from transformer_config import BATCH_SIZE, MAX_LENGTH

"""
Embedding backends for transformer retrieval.

Supports:
- Hugging Face encoder models + mean pooling
- SentenceTransformer models + .encode()
"""


def get_device() -> str:
    """Return CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each embedding vector to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def make_safe_model_name(model_name: str) -> str:
    """Convert model name into a filesystem-safe string."""
    return model_name.replace("/", "_")


def get_model_type(model_name: str) -> str:
    """
    Route model names to the correct embedding backend.
    """
    if model_name.startswith("sentence-transformers/"):
        return "sentence_transformer"
    return "hf_encoder"


def hash_text(text: str) -> str:
    """Create a stable hash for patient embedding cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_hf_model_and_tokenizer(model_name: str):
    """
    Load a Hugging Face encoder model and BERT-family tokenizer.
    """
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.to(get_device())
    model.eval()
    return tokenizer, model


def mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings while ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts_hf(
    tokenizer,
    model,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Embed texts with a Hugging Face encoder and mean pooling.
    """
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
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy())

    matrix = np.vstack(embeddings).astype(np.float32)
    return l2_normalize(matrix)


def load_sentence_transformer_model(model_name: str):
    """Load a SentenceTransformer model."""
    return SentenceTransformer(model_name, device=get_device())


def embed_texts_sentence_transformer(
    model,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed texts with a SentenceTransformer model.
    """
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
    """
    Load the correct embedding backend for a model.
    """
    model_type = get_model_type(model_name)

    if model_type == "hf_encoder":
        tokenizer, model = load_hf_model_and_tokenizer(model_name)
        return {
            "model_type": model_type,
            "tokenizer": tokenizer,
            "model": model,
        }

    if model_type == "sentence_transformer":
        model = load_sentence_transformer_model(model_name)
        return {
            "model_type": model_type,
            "model": model,
        }

    raise ValueError(f"Unsupported model type: {model_name}")


def embed_texts(backend: dict, texts: List[str]) -> np.ndarray:
    """
    Dispatch embedding calls to the appropriate backend.
    """
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
