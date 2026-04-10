import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoModel, BertTokenizer, BertTokenizerFast
from sentence_transformers import SentenceTransformer

"""
Transformer-based disease retrieval pipeline.

What this script does:
1. Build text representations for patients and diseases
2. Embed those texts with either:
   - Hugging Face encoder models + mean pooling
   - SentenceTransformer models + .encode()
3. Compare patient embedding against all disease embeddings with cosine similarity
4. Return the top-ranked diseases

- We compute scores on all disease IDs that exist in disease_profiles.json
- Then we collapse alias IDs (OMIM, MONDO, ORPHA, etc.) into canonical diseases
  using alias_to_canonical.json
- This avoids showing the same disease multiple times in the final ranking
"""


# =========================================================
# Config
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
TRANSFORMER_DIR = PROJECT_ROOT / "outputs" / "transformer"
CACHE_ROOT = TRANSFORMER_DIR / "cache"

TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_LIST = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    # "dmis-lab/biobert-base-cased-v1.1",
    # "emilyalsentzer/Bio_ClinicalBERT",
    # "sentence-transformers/all-MiniLM-L6-v2",
]

TOP_K = 10
MAX_LENGTH = 128
BATCH_SIZE = 16

DISEASE_PROFILES_PATH = SHARED_DIR / "disease_profiles.json"
HPO_LABELS_PATH = SHARED_DIR / "hpo_labels.json"
PATIENT_PATH = SHARED_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = SHARED_DIR / "alias_to_canonical.json"


# =========================================================
# Basic I/O
# =========================================================
def load_json(path: Path) -> Any:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save Python data as formatted JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# Utilities
# =========================================================
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def unique_preserve_order(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving first occurrence order.

    Used mainly for phenotype labels so repeated HPO labels do not create
    redundant text segments.
    """
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """ Normalize each embedding vector to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def make_safe_model_name(model_name: str) -> str:
    """Convert a Hugging Face model name into a filesystem-safe name."""
    return model_name.replace("/", "_")


def get_model_type(model_name: str) -> str:
    """
    Route a model name to the correct embedding backend.

    Current logic:
    - sentence-transformers/... -> SentenceTransformer encode pipeline
    - everything else -> Hugging Face encoder + mean pooling
    """
    if model_name.startswith("sentence-transformers/"):
        return "sentence_transformer"
    return "hf_encoder"


def hash_text(text: str) -> str:
    """
    Create a stable hash for text.

    Used for in-memory patient embedding cache, so repeated identical queries
    do not recompute the patient embedding.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =========================================================
# Text construction
# =========================================================
def hpo_terms_to_labels(hpo_terms: List[str], hpo_labels: Dict[str, str]) -> List[str]:
    """
    Convert HPO IDs into human-readable labels.

    Example:
    HP:0001263 -> developmental delay

    We only keep labels that exist in hpo_labels and remove duplicates.
    """
    labels = []
    for term in hpo_terms:
        label = hpo_labels.get(term)
        if label:
            labels.append(label.strip())
    return unique_preserve_order(labels)


def build_patient_text(patient: dict, hpo_labels: Dict[str, str]) -> str:
    """
    Build the text that will represent the patient for the embedding model.

    We combine:
    - raw clinical description, if available
    - HPO phenotype labels

    This creates a hybrid text representation rather than relying only on raw text.
    """
    raw_text = (patient.get("raw_text") or "").strip()
    hpo_terms = patient.get("hpo_terms", [])
    phenotype_labels = hpo_terms_to_labels(hpo_terms, hpo_labels)

    parts = []

    if raw_text:
        parts.append(f"Patient description: {raw_text}")

    if phenotype_labels:
        parts.append(f"Patient phenotypes: {'; '.join(phenotype_labels)}")

    return " ".join(parts).strip()


def build_disease_text(profile: dict, hpo_labels: Dict[str, str]) -> str:
    """
    Build the text that will represent a disease for the embedding model.

    We combine:
    - disease label
    - merged free-text description
    - HPO phenotype labels

    The result is a single text block that the encoder can embed.
    """
    label = (profile.get("label") or "").strip()
    desc = (profile.get("merged_description") or "").strip()
    hpo_terms = profile.get("hpo_terms", [])
    phenotype_labels = hpo_terms_to_labels(hpo_terms, hpo_labels)

    parts = []

    if label:
        parts.append(f"Disease: {label}")

    if desc:
        parts.append(f"Description: {desc}")

    if phenotype_labels:
        parts.append(f"Phenotypes: {'; '.join(phenotype_labels)}")

    return " ".join(parts).strip()


def build_disease_texts(
    disease_profiles: Dict[str, dict],
    hpo_labels: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Precompute the disease texts for all disease profiles.

    Returns parallel lists:
    - disease_ids
    - disease_labels
    - disease_texts

    These stay aligned by index throughout the pipeline.
    """
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


# =========================================================
# HF encoder pipeline
# =========================================================
def load_hf_model_and_tokenizer(model_name: str):
    """Load a Hugging Face encoder model and tokenizer."""
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.to(get_device())
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply attention-mask-aware mean pooling.

    Why:
    - Transformer models output token-level embeddings
    - We need one vector per text
    - Mean pooling averages only the real tokens, not the padding
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
    Embed texts with a Hugging Face encoder model.

    Steps:
    1. Tokenize in batches
    2. Run the encoder
    3. Mean-pool token embeddings
    4. L2-normalize final vectors
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


# =========================================================
# SentenceTransformer pipeline
# =========================================================
def load_sentence_transformer_model(model_name: str):
    """
    Load a SentenceTransformer model.

    SentenceTransformer models already handle text pooling internally.
    """
    return SentenceTransformer(model_name, device=get_device())


def embed_texts_sentence_transformer(
    model,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed texts with a SentenceTransformer model.

    We request normalized embeddings directly because these models are designed
    for similarity search and sentence-level embeddings.
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


# =========================================================
# Unified embedding backend
# =========================================================
def load_embedding_backend(model_name: str) -> dict:
    """
    Load the correct embedding backend for a model.

    This hides backend-specific details from the rest of the pipeline.
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
    Dispatch embedding calls to the correct backend implementation.
    """
    model_type = backend["model_type"]

    if model_type == "hf_encoder":
        return embed_texts_hf(
            tokenizer=backend["tokenizer"],
            model=backend["model"],
            texts=texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )

    if model_type == "sentence_transformer":
        return embed_texts_sentence_transformer(
            model=backend["model"],
            texts=texts,
            batch_size=BATCH_SIZE,
        )

    raise ValueError(f"Unsupported backend type: {model_type}")


# =========================================================
# Persistent cache handling
# =========================================================
def get_cache_paths(model_name: str) -> Dict[str, Path]:
    """
    Return all cache file paths for one model.

    Each model gets its own cache directory because embeddings from different
    models are not interchangeable.
    """
    safe_name = make_safe_model_name(model_name)
    model_cache_dir = CACHE_ROOT / safe_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "ids": model_cache_dir / "disease_ids.json",
        "labels": model_cache_dir / "disease_labels.json",
        "texts": model_cache_dir / "disease_texts.json",
        "embeddings": model_cache_dir / "disease_embeddings.npy",
    }


def persistent_cache_exists(cache_paths: Dict[str, Path]) -> bool:
    """
    Check whether the full on-disk embedding cache exists for a model.
    """
    return (
        cache_paths["ids"].exists()
        and cache_paths["labels"].exists()
        and cache_paths["texts"].exists()
        and cache_paths["embeddings"].exists()
    )


def load_persistent_cache(cache_paths: Dict[str, Path]) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Load disease IDs, labels, texts, and embeddings from disk cache.
    """
    disease_ids = load_json(cache_paths["ids"])
    disease_labels = load_json(cache_paths["labels"])
    disease_texts = load_json(cache_paths["texts"])
    disease_embeddings = np.load(cache_paths["embeddings"])
    return disease_ids, disease_labels, disease_texts, disease_embeddings


def save_persistent_cache(
    cache_paths: Dict[str, Path],
    disease_ids: List[str],
    disease_labels: List[str],
    disease_texts: List[str],
    disease_embeddings: np.ndarray,
) -> None:
    """
    Save disease metadata and embeddings to disk so future runs do not
    recompute disease vectors.
    """
    save_json(disease_ids, cache_paths["ids"])
    save_json(disease_labels, cache_paths["labels"])
    save_json(disease_texts, cache_paths["texts"])
    np.save(cache_paths["embeddings"], disease_embeddings)


# =========================================================
# Canonicalization / deduplication helpers
# =========================================================
def collapse_ranked_results_to_canonical(
    ranked_indices: np.ndarray,
    scores: np.ndarray,
    disease_ids: List[str],
    disease_labels: List[str],
    disease_texts: List[str],
    alias_to_canonical: Dict[str, str],
    model_name: str,
    model_type: str,
    patient_text: str,
    top_k: int,
) -> List[dict]:
    """
    Collapse alias-level matches into canonical disease-level results.

    Why this matters:
    - A disease may appear under multiple IDs (OMIM, MONDO, ORPHA, etc.)
    - Without deduplication, the same disease can occupy several top ranks
    - This function keeps only the best-scoring alias per canonical disease

    Output logic:
    - representative_disease_id = alias with highest score
    - canonical_disease_id = canonical ID from alias_to_canonical
    - matched_aliases = all alias IDs that appeared for this canonical disease
    """
    grouped = {}

    for idx in ranked_indices:
        disease_id = disease_ids[idx]
        canonical_id = alias_to_canonical.get(disease_id, disease_id)
        score = float(scores[idx])

        if canonical_id not in grouped:
            grouped[canonical_id] = {
                "canonical_disease_id": canonical_id,
                "representative_disease_id": disease_id,
                "label": disease_labels[idx],
                "score": score,
                "matched_aliases": [disease_id],
                "disease_text_preview": disease_texts[idx][:300],
            }
        else:
            grouped[canonical_id]["matched_aliases"].append(disease_id)

            # Keep the highest-scoring alias as the representative row
            if score > grouped[canonical_id]["score"]:
                grouped[canonical_id]["representative_disease_id"] = disease_id
                grouped[canonical_id]["label"] = disease_labels[idx]
                grouped[canonical_id]["score"] = score
                grouped[canonical_id]["disease_text_preview"] = disease_texts[idx][:300]

    # Sort canonical groups by their best score
    collapsed = sorted(
        grouped.values(),
        key=lambda row: row["score"],
        reverse=True,
    )[:top_k]

    results = []
    for rank_idx, row in enumerate(collapsed, start=1):
        results.append(
            {
                "rank": rank_idx,
                "canonical_disease_id": row["canonical_disease_id"],
                "representative_disease_id": row["representative_disease_id"],
                "label": row["label"],
                "model_name": model_name,
                "model_type": model_type,
                "score": row["score"],
                "matched_aliases": sorted(set(row["matched_aliases"])),
                "explanation": {
                    "patient_text_preview": patient_text[:300],
                    "disease_text_preview": row["disease_text_preview"],
                },
                "metadata": {
                    "top_k": top_k,
                    "embedding_normalization": "l2",
                    "max_length": MAX_LENGTH if model_type == "hf_encoder" else None,
                    "pooling": "mean" if model_type == "hf_encoder" else None,
                    "sentence_transformer_encode": model_type == "sentence_transformer",
                    "deduplicated_to_canonical": True,
                },
            }
        )

    return results


# =========================================================
# In-memory registry
# =========================================================
class DiseaseRetriever:
    """
    Main retrieval object.

    Responsibilities:
    - hold disease metadata
    - load models on demand
    - cache disease embeddings on disk
    - cache patient embeddings in memory
    - rank diseases for a patient query
    """

    def __init__(
        self,
        disease_profiles: Dict[str, dict],
        hpo_labels: Dict[str, str],
        alias_to_canonical: Dict[str, str],
        model_list: List[str],
        rebuild_cache: bool = False,
    ):
        self.disease_profiles = disease_profiles
        self.hpo_labels = hpo_labels
        self.alias_to_canonical = alias_to_canonical
        self.model_list = model_list
        self.rebuild_cache = rebuild_cache

        # Precompute disease texts once because these are static.
        self.global_disease_ids, self.global_disease_labels, self.global_disease_texts = build_disease_texts(
            disease_profiles=self.disease_profiles,
            hpo_labels=self.hpo_labels,
        )

        # Backends stay in memory after first load.
        self.backends: Dict[str, dict] = {}

        # Holds loaded disease embeddings and associated metadata per model.
        self.model_registry: Dict[str, dict] = {}

        # Optional cache for repeated identical patient queries.
        self.patient_embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def warmup(self, preload_models: bool = True) -> None:
        """Preload model resources."""
        for model_name in self.model_list:
            if preload_models:
                backend = self._get_backend(model_name)
            else:
                backend = None

            self._ensure_model_resources(model_name, backend=backend)

    def _get_backend(self, model_name: str) -> dict:
        """
        Load one model backend lazily and keep it in memory.
        """
        if model_name not in self.backends:
            self.backends[model_name] = load_embedding_backend(model_name)
        return self.backends[model_name]

    def _ensure_model_resources(self, model_name: str, backend: dict | None = None) -> None:
        """
        Ensure disease embeddings for one model are available in memory.

        Logic:
        - If already loaded, reuse
        - Else try disk cache
        - Else compute embeddings, save cache, then load into registry
        """
        if model_name in self.model_registry and not self.rebuild_cache:
            return

        cache_paths = get_cache_paths(model_name)

        if persistent_cache_exists(cache_paths) and not self.rebuild_cache:
            disease_ids, disease_labels, disease_texts, disease_embeddings = load_persistent_cache(cache_paths)
        else:
            if backend is None:
                backend = self._get_backend(model_name)

            disease_ids = self.global_disease_ids
            disease_labels = self.global_disease_labels
            disease_texts = self.global_disease_texts
            disease_embeddings = embed_texts(backend, disease_texts)

            save_persistent_cache(
                cache_paths=cache_paths,
                disease_ids=disease_ids,
                disease_labels=disease_labels,
                disease_texts=disease_texts,
                disease_embeddings=disease_embeddings,
            )

        self.model_registry[model_name] = {
            "model_type": get_model_type(model_name),
            "disease_ids": disease_ids,
            "disease_labels": disease_labels,
            "disease_texts": disease_texts,
            "disease_embeddings": disease_embeddings,
            "cache_paths": cache_paths,
        }

    def _get_patient_embedding(self, model_name: str, patient_text: str) -> np.ndarray:
        """
        Return the patient embedding, using in-memory cache when possible.

        Since many repeated queries may reuse the same patient text,
        this avoids recomputing the same embedding.
        """
        text_hash = hash_text(patient_text)
        cache_key = (model_name, text_hash)

        if cache_key in self.patient_embedding_cache:
            return self.patient_embedding_cache[cache_key]

        backend = self._get_backend(model_name)
        patient_embedding = embed_texts(backend, [patient_text])[0]
        self.patient_embedding_cache[cache_key] = patient_embedding
        return patient_embedding

    def rank(
        self,
        model_name: str,
        patient: dict,
        top_k: int = TOP_K,
        candidate_pool_size: int = 200,
    ) -> List[dict]:
        """
        Rank diseases for one patient query.

        Important detail:
        - We first rank all alias-level disease entries
        - Then we collapse them to unique canonical diseases
        - candidate_pool_size should be larger than top_k so deduplication still
          leaves enough unique canonical candidates
        """
        if model_name not in self.model_list:
            raise ValueError(f"Model not available: {model_name}")

        self._ensure_model_resources(model_name)

        patient_text = build_patient_text(patient, self.hpo_labels)
        if not patient_text:
            raise ValueError("Patient text is empty.")

        patient_embedding = self._get_patient_embedding(model_name, patient_text)

        resources = self.model_registry[model_name]
        disease_ids = resources["disease_ids"]
        disease_labels = resources["disease_labels"]
        disease_texts = resources["disease_texts"]
        disease_embeddings = resources["disease_embeddings"]
        model_type = resources["model_type"]

        # Because embeddings are L2-normalized, dot product = cosine similarity.
        scores = disease_embeddings @ patient_embedding

        # We retrieve a larger pool first, then deduplicate by canonical disease.
        pool_size = min(candidate_pool_size, len(scores))
        ranked_indices = np.argsort(-scores)[:pool_size]

        results = collapse_ranked_results_to_canonical(
            ranked_indices=ranked_indices,
            scores=scores,
            disease_ids=disease_ids,
            disease_labels=disease_labels,
            disease_texts=disease_texts,
            alias_to_canonical=self.alias_to_canonical,
            model_name=model_name,
            model_type=model_type,
            patient_text=patient_text,
            top_k=top_k,
        )

        return results


# =========================================================
# Main
# =========================================================
def main():
    disease_profiles = load_json(DISEASE_PROFILES_PATH)
    hpo_labels = load_json(HPO_LABELS_PATH)
    patient = load_json(PATIENT_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)

    retriever = DiseaseRetriever(
        disease_profiles=disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=MODEL_LIST,
        rebuild_cache=False,
    )

    retriever.warmup(preload_models=True)

    all_results = {}

    for model_name in MODEL_LIST:
        print(f"\nRunning model: {model_name}")
        print(f"Model type: {get_model_type(model_name)}")

        results = retriever.rank(
            model_name=model_name,
            patient=patient,
            top_k=TOP_K,
            candidate_pool_size=200,
        )

        safe_name = make_safe_model_name(model_name)
        out_path = TRANSFORMER_DIR / f"{safe_name}_top{TOP_K}_canonical.json"
        save_json(results, out_path)

        all_results[model_name] = results

        for r in results:
            print(
                f"rank={r['rank']:>2} | "
                f"{r['canonical_disease_id']:<15} | "
                f"score={r['score']:.4f} | "
                f"{r['label']} | "
                f"aliases={len(r['matched_aliases'])}"
            )

        print(f"Saved to: {out_path}")

    summary_path = TRANSFORMER_DIR / "all_model_results_summary_canonical.json"
    save_json(all_results, summary_path)
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
