"""
DiseaseRetriever — transformer-based disease ranking.

Handles:
- persistent embedding cache (per model, on disk)
- canonical deduplication (alias → canonical disease ID)
- patient embedding caching (in-memory, by text hash)

Uses methods.py for text construction and embedding backends.
Uses shared.io for all file I/O.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from shared.io import load_json, save_json
from similarity_methods.transformer.config import CACHE_ROOT, MAX_LENGTH, TOP_K
from similarity_methods.transformer.methods import (
    build_disease_texts,
    build_patient_text,
    embed_texts,
    get_model_type,
    hash_text,
    load_embedding_backend,
    make_safe_model_name,
)


# ── Cache utilities ───────────────────────────────────────────────────────────


def get_cache_paths(model_name: str) -> Dict[str, Path]:
    """Return cache file paths for one model."""
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
    """Check if the full embedding cache exists for one model."""
    return all(cache_paths[k].exists() for k in ["ids", "labels", "texts", "embeddings"])


def load_persistent_cache(
    cache_paths: Dict[str, Path],
) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """Load disease IDs, labels, texts, and embeddings from disk cache."""
    return (
        load_json(cache_paths["ids"]),
        load_json(cache_paths["labels"]),
        load_json(cache_paths["texts"]),
        np.load(cache_paths["embeddings"]),
    )


def save_persistent_cache(
    cache_paths: Dict[str, Path],
    disease_ids: List[str],
    disease_labels: List[str],
    disease_texts: List[str],
    disease_embeddings: np.ndarray,
) -> None:
    """Save disease metadata and embeddings to disk."""
    save_json(disease_ids, cache_paths["ids"])
    save_json(disease_labels, cache_paths["labels"])
    save_json(disease_texts, cache_paths["texts"])
    np.save(cache_paths["embeddings"], disease_embeddings)


# ── Canonical deduplication ───────────────────────────────────────────────────


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
    Collapse alias-level results into canonical disease-level results.

    When multiple aliases of the same canonical disease appear in the
    ranked list, keeps the highest-scoring alias as representative
    and collects all matched aliases.
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
            if score > grouped[canonical_id]["score"]:
                grouped[canonical_id]["representative_disease_id"] = disease_id
                grouped[canonical_id]["label"] = disease_labels[idx]
                grouped[canonical_id]["score"] = score
                grouped[canonical_id]["disease_text_preview"] = disease_texts[idx][:300]

    collapsed = sorted(
        grouped.values(), key=lambda row: row["score"], reverse=True
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


# ── DiseaseRetriever ──────────────────────────────────────────────────────────


class DiseaseRetriever:
    """
    Main retrieval object for transformer-based disease ranking.

    Handles model loading, embedding caching, and ranking.
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

        (
            self.global_disease_ids,
            self.global_disease_labels,
            self.global_disease_texts,
        ) = build_disease_texts(
            disease_profiles=self.disease_profiles,
            hpo_labels=self.hpo_labels,
        )

        self.backends: Dict[str, dict] = {}
        self.model_registry: Dict[str, dict] = {}
        self.patient_embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def warmup(self, preload_models: bool = True) -> None:
        """Preload all models and build/load embedding caches."""
        for model_name in self.model_list:
            backend = self._get_backend(model_name) if preload_models else None
            self._ensure_model_resources(model_name, backend=backend)

    def _get_backend(self, model_name: str) -> dict:
        if model_name not in self.backends:
            self.backends[model_name] = load_embedding_backend(model_name)
        return self.backends[model_name]

    def _ensure_model_resources(
        self,
        model_name: str,
        backend: dict | None = None,
    ) -> None:
        if model_name in self.model_registry and not self.rebuild_cache:
            return

        cache_paths = get_cache_paths(model_name)

        if persistent_cache_exists(cache_paths) and not self.rebuild_cache:
            disease_ids, disease_labels, disease_texts, disease_embeddings = (
                load_persistent_cache(cache_paths)
            )
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

    def _get_patient_embedding(
        self,
        model_name: str,
        patient_text: str,
    ) -> np.ndarray:
        """Get patient embedding, using in-memory cache if available."""
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
        """Rank diseases for a patient using the specified model."""
        if model_name not in self.model_list:
            raise ValueError(f"Model not available: {model_name}")

        self._ensure_model_resources(model_name)

        patient_text = build_patient_text(patient, self.hpo_labels)
        if not patient_text:
            raise ValueError("Patient text is empty.")

        patient_embedding = self._get_patient_embedding(model_name, patient_text)

        resources = self.model_registry[model_name]
        scores = resources["disease_embeddings"] @ patient_embedding
        pool_size = min(candidate_pool_size, len(scores))
        ranked_indices = np.argsort(-scores)[:pool_size]

        return collapse_ranked_results_to_canonical(
            ranked_indices=ranked_indices,
            scores=scores,
            disease_ids=resources["disease_ids"],
            disease_labels=resources["disease_labels"],
            disease_texts=resources["disease_texts"],
            alias_to_canonical=self.alias_to_canonical,
            model_name=model_name,
            model_type=resources["model_type"],
            patient_text=patient_text,
            top_k=top_k,
        )
    