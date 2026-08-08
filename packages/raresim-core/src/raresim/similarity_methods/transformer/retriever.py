"""
DiseaseRetriever — transformer-based disease ranking.

Handles:
- persistent embedding cache (per model, on disk)
- canonical deduplication (alias → canonical disease ID)
- patient embedding caching (in-memory, by text hash)
- shared phenotype label explanation per result

Uses methods.py for text construction and embedding backends.
Uses shared.io for all file I/O.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from raresim.core.context import AppContext
from raresim.core.pipeline import build_run_stats
from raresim.ontology.disease_category import build_category_metadata
from raresim.similarity_methods.transformer.config import (
    CACHE_ROOT,
    MAX_LENGTH,
    TEXT_PREVIEW_LENGTH,
)
from raresim.similarity_methods.transformer.explanation import (
    build_explanation,
    build_method_specific_explanation_block,
)
from raresim.similarity_methods.transformer.methods import (
    build_disease_texts,
    build_patient_text,
    embed_texts,
    get_disease_terms,
    get_model_type,
    hash_text,
    load_embedding_backend,
    select_phenotype_labels,
)
from raresim.types.result import RunStats, SimilarityResult
from raresim.types.schemas import PatientProfile
from raresim.utils.io import load_json, make_safe_model_name, save_json

# ── Cache utilities ───────────────────────────────────────────────────────────


def get_cache_paths(model_name: str) -> dict[str, Path]:
    """Return cache file paths for one model."""
    safe_name = make_safe_model_name(model_name)
    model_cache_dir = CACHE_ROOT / safe_name
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "ids": model_cache_dir / "disease_ids.json",
        "labels": model_cache_dir / "disease_labels.json",
        "texts": model_cache_dir / "disease_texts.json",
        "embeddings": model_cache_dir / "disease_embeddings.npy",
        "metadata": model_cache_dir / "metadata.json",
    }


def persistent_cache_exists(cache_paths: dict[str, Path]) -> bool:
    """Check if the full embedding cache exists for one model."""
    return all(
        cache_paths[key].exists()
        for key in ["ids", "labels", "texts", "embeddings", "metadata"]
    )


def compute_disease_fingerprint(
    disease_ids: list[str], disease_texts: list[str]
) -> str:
    """
    Stable hash of disease IDs and embedding texts.

    Used to detect stale caches when the disease set or text-building logic
    changes.
    """
    hasher = hashlib.sha256()
    for disease_id, text in zip(disease_ids, disease_texts):
        hasher.update(disease_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def build_cache_metadata(
    *,
    model_name: str,
    disease_ids: list[str],
    disease_texts: list[str],
) -> dict:
    """Build the metadata record written/checked alongside a disease cache."""
    return {
        "model_name": model_name,
        "model_type": get_model_type(model_name),
        "max_length": MAX_LENGTH,
        "n_diseases": len(disease_ids),
        "fingerprint": compute_disease_fingerprint(disease_ids, disease_texts),
    }


def cache_metadata_matches(cache_paths: dict[str, Path], expected: dict) -> bool:
    """
    Return True only if on-disk cache metadata matches the current run.

    Missing or unreadable metadata counts as a mismatch and forces a rebuild.
    """
    try:
        stored = load_json(cache_paths["metadata"])
    except (FileNotFoundError, ValueError, OSError):
        return False

    return stored == expected


def load_json_string_list(input_path: Path) -> list[str]:
    """Load a JSON file that should contain a list of strings."""
    data = load_json(input_path)

    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {input_path}, got {type(data).__name__}")

    return [str(item) for item in data]


def load_persistent_cache(
    cache_paths: dict[str, Path],
) -> tuple[list[str], list[str], list[str], np.ndarray]:
    """Load disease IDs, labels, texts, and embeddings from disk cache."""
    return (
        load_json_string_list(cache_paths["ids"]),
        load_json_string_list(cache_paths["labels"]),
        load_json_string_list(cache_paths["texts"]),
        np.load(cache_paths["embeddings"]),
    )


def save_persistent_cache(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    cache_paths: dict[str, Path],
    disease_ids: list[str],
    disease_labels: list[str],
    disease_texts: list[str],
    disease_embeddings: np.ndarray,
    metadata: dict,
) -> None:
    """Save disease metadata, embeddings, and cache-validity metadata to disk."""
    save_json(disease_ids, cache_paths["ids"])
    save_json(disease_labels, cache_paths["labels"])
    save_json(disease_texts, cache_paths["texts"])
    np.save(cache_paths["embeddings"], disease_embeddings)
    save_json(metadata, cache_paths["metadata"])


# ── Canonical deduplication ───────────────────────────────────────────────────


def _get_result_profile(
    canonical_id: str,
    representative_id: str,
    disease_profiles: dict[str, dict],
) -> dict:
    """
    Return the best available disease profile.

    Prefer the canonical profile. Fall back to the representative alias profile.
    """
    return (
        disease_profiles.get(canonical_id)
        or disease_profiles.get(representative_id)
        or {}
    )


def select_top_canonical_indices(
    scores: np.ndarray,
    disease_ids: list[str],
    alias_to_canonical: dict[str, str],
    pool_size: int,
) -> np.ndarray:
    """
    Select alias-level indices such that `pool_size` distinct CANONICAL
    diseases survive the pool, not `pool_size` alias-level rows.
    """
    best_idx_per_canonical: dict[str, int] = {}
    for idx, disease_id in enumerate(disease_ids):
        canonical_id = alias_to_canonical.get(disease_id, disease_id)
        current_best = best_idx_per_canonical.get(canonical_id)
        if current_best is None or scores[idx] > scores[current_best]:
            best_idx_per_canonical[canonical_id] = idx

    canonical_ids = list(best_idx_per_canonical.keys())
    best_indices = np.array([best_idx_per_canonical[c] for c in canonical_ids])
    canonical_scores = scores[best_indices]

    top_n = min(pool_size, len(canonical_ids))
    top_order = np.argpartition(-canonical_scores, top_n - 1)[:top_n]
    selected_canonical_ids = {canonical_ids[i] for i in top_order}

    # Return every alias belonging to a selected canonical id, so
    # collapse_ranked_results_to_canonical still sees all matched aliases.
    return np.array(
        [
            idx
            for idx, disease_id in enumerate(disease_ids)
            if alias_to_canonical.get(disease_id, disease_id) in selected_canonical_ids
        ]
    )


def collapse_ranked_results_to_canonical(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    ranked_indices: np.ndarray,
    scores: np.ndarray,
    disease_ids: list[str],
    disease_labels: list[str],
    disease_texts: list[str],
    disease_profiles: dict[str, dict],
    alias_to_canonical: dict[str, str],
    disease_ancestors: dict[str, list[str]],
    disease_metadata_index: dict[str, dict],
    model_name: str,
    model_type: str,
    patient_text: str,
    top_k: int,
    patient_hpo_terms: list[str],
    hpo_labels: dict[str, str],
    ic_values: dict[str, float],
    result_method_name: str,
    patient: PatientProfile,
) -> list[SimilarityResult]:
    """
    Collapse alias-level results into canonical disease-level results.

    When multiple aliases of the same canonical disease appear in the ranked
    list, the highest-scoring alias is kept as representative and all matched
    aliases are preserved.
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
                "disease_text_preview": disease_texts[idx][:TEXT_PREVIEW_LENGTH],
            }
            continue

        grouped[canonical_id]["matched_aliases"].append(disease_id)

        if score > grouped[canonical_id]["score"]:
            grouped[canonical_id]["representative_disease_id"] = disease_id
            grouped[canonical_id]["label"] = disease_labels[idx]
            grouped[canonical_id]["score"] = score
            grouped[canonical_id]["disease_text_preview"] = disease_texts[idx][
                :TEXT_PREVIEW_LENGTH
            ]

    collapsed = sorted(
        grouped.values(),
        key=lambda row: row["score"],
        reverse=True,
    )[:top_k]

    results = []

    for rank_idx, row in enumerate(collapsed, start=1):
        canonical_id = row["canonical_disease_id"]
        representative_id = row["representative_disease_id"]

        disease_profile = _get_result_profile(
            canonical_id=canonical_id,
            representative_id=representative_id,
            disease_profiles=disease_profiles,
        )
        disease_hpo_terms = get_disease_terms(disease_profile)

        category_metadata = build_category_metadata(
            disease_id=canonical_id,
            profile=disease_profile,
            disease_ancestors=disease_ancestors,
            disease_metadata_index=disease_metadata_index,
        )

        method_specific_block = build_method_specific_explanation_block(
            method_name=result_method_name,
            model_name=model_name,
            model_type=model_type,
            patient_text=patient_text,
            disease_text_preview=row["disease_text_preview"],
        )

        results.append(
            SimilarityResult(
                disease_id=canonical_id,
                label=row["label"],
                score=row["score"],
                method_name=result_method_name,
                profile_type=category_metadata["profile_type"],
                category_source_id=category_metadata["category_source_id"],
                category_path=category_metadata["category_path"],
                # Aliases actually collapsed into this canonical result during
                # this ranking pass, not all known aliases for the category.
                matched_aliases=row["matched_aliases"],
                rank=rank_idx,
                explanation=build_explanation(
                    score=row["score"],
                    model_name=model_name,
                    patient_hpo_terms=patient_hpo_terms,
                    disease_hpo_terms=disease_hpo_terms,
                    ic_values=ic_values,
                    hpo_labels=hpo_labels,
                    method_specific=method_specific_block,
                    patient=patient,
                ),
            )
        )

    return results


# ── DiseaseRetriever ──────────────────────────────────────────────────────────


class DiseaseRetriever:  # pylint: disable=too-many-instance-attributes
    """
    Main retrieval object for transformer-based disease ranking.

    Handles model loading, disease embedding caching, and ranking.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        disease_profiles: dict[str, dict],
        hpo_labels: dict[str, str],
        alias_to_canonical: dict[str, str],
        model_list: list[str],
        patient: PatientProfile,
        *,
        disease_ancestors: dict[str, list[str]] | None = None,
        disease_metadata_index: dict[str, dict] | None = None,
        rebuild_cache: bool = False,
        ic_values: dict[str, float] | None = None,
    ) -> None:
        self.disease_profiles = disease_profiles
        self.hpo_labels = hpo_labels
        self.alias_to_canonical = alias_to_canonical
        self.model_list = model_list
        self.patient = patient
        self.disease_ancestors = disease_ancestors or {}
        self.disease_metadata_index = disease_metadata_index or {}
        self.rebuild_cache = rebuild_cache
        self.ic_values = ic_values or {}

        (
            self.global_disease_ids,
            self.global_disease_labels,
            self.global_disease_texts,
        ) = build_disease_texts(
            disease_profiles=self.disease_profiles,
            hpo_labels=self.hpo_labels,
        )

        self.backends: dict[str, dict] = {}
        self.model_registry: dict[str, dict] = {}
        self.patient_embedding_cache: dict[tuple[str, str], np.ndarray] = {}

    @classmethod
    def from_context(
        cls,
        patient: PatientProfile,
        ctx: AppContext,
        model_list: list[str],
        *,
        rebuild_cache: bool = False,
    ) -> "DiseaseRetriever":
        """Create a DiseaseRetriever from an AppContext."""
        return cls(
            patient=patient,
            disease_profiles=ctx.disease_profiles,
            hpo_labels=ctx.hpo_labels,
            alias_to_canonical=ctx.alias_to_canonical,
            model_list=model_list,
            disease_ancestors=ctx.disease_ancestors,
            disease_metadata_index=ctx.disease_metadata_index,
            rebuild_cache=rebuild_cache,
            ic_values=ctx.ic_values,
        )

    def warmup(self, preload_models: bool = False) -> None:
        """
        Prepare disease embedding resources.

        If preload_models is False, existing persistent caches are loaded without
        loading the transformer backend. The backend is loaded only when a cache
        is missing or when patient embedding is needed during ranking.
        """
        for model_name in self.model_list:
            print(f"  Preparing: {model_name}")
            backend = self._get_backend(model_name) if preload_models else None
            self._ensure_model_resources(model_name, backend=backend)
            if not preload_models:
                self.unload_backend(model_name)

    def _get_backend(self, model_name: str) -> dict:
        """Load or return an embedding backend for a model."""
        if model_name not in self.backends:
            self.backends[model_name] = load_embedding_backend(model_name)

        return self.backends[model_name]

    def unload_backend(self, model_name: str) -> None:
        """Unload one model backend and release CUDA cache if possible."""
        backend = self.backends.pop(model_name, None)
        if backend is None:
            return

        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_model_resources(
        self,
        model_name: str,
        backend: dict | None = None,
    ) -> None:
        """Ensure disease embeddings exist for the selected model."""
        if model_name in self.model_registry:
            return

        cache_paths = get_cache_paths(model_name)
        expected_metadata = build_cache_metadata(
            model_name=model_name,
            disease_ids=self.global_disease_ids,
            disease_texts=self.global_disease_texts,
        )

        cache_is_valid = (
            persistent_cache_exists(cache_paths)
            and not self.rebuild_cache
            and cache_metadata_matches(cache_paths, expected_metadata)
        )

        if cache_is_valid:
            print(f"    Loading from cache: {model_name}")
            disease_ids, disease_labels, disease_texts, disease_embeddings = (
                load_persistent_cache(cache_paths)
            )
        else:
            if persistent_cache_exists(cache_paths) and not self.rebuild_cache:
                print(
                    "    Cache exists but metadata does not match current "
                    f"config or disease text; rebuilding: {model_name}"
                )
            print(f"    Building embedding cache: {model_name}")

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
                metadata=expected_metadata,
            )
            print(f"    Cache saved: {cache_paths['embeddings']}")

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
        patient: PatientProfile | None = None,
        top_k: int = 20,
        candidate_pool_size: int = 200,
    ) -> list[SimilarityResult]:
        """Rank diseases for a patient using one model.

        The optional patient argument is kept for backward compatibility with
        older callers. If provided, it becomes the active patient for run_stats.
        """
        if model_name not in self.model_list:
            raise ValueError(f"Model not available: {model_name}")

        if patient is not None:
            self.patient = patient

        self._ensure_model_resources(model_name)

        patient_text = build_patient_text(self.patient, self.hpo_labels)
        if not patient_text:
            raise ValueError("Patient text is empty.")

        patient_hpo_terms = self.patient.get_terms(use_propagated=False)
        patient_embedding = self._get_patient_embedding(model_name, patient_text)

        resources = self.model_registry[model_name]
        scores = resources["disease_embeddings"] @ patient_embedding
        pool_size = min(candidate_pool_size, len(scores))
        ranked_indices = select_top_canonical_indices(
            scores=scores,
            disease_ids=resources["disease_ids"],
            alias_to_canonical=self.alias_to_canonical,
            pool_size=pool_size,
        )

        return collapse_ranked_results_to_canonical(
            ranked_indices=ranked_indices,
            scores=scores,
            disease_ids=resources["disease_ids"],
            disease_labels=resources["disease_labels"],
            disease_texts=resources["disease_texts"],
            disease_profiles=self.disease_profiles,
            alias_to_canonical=self.alias_to_canonical,
            disease_ancestors=self.disease_ancestors,
            disease_metadata_index=self.disease_metadata_index,
            model_name=model_name,
            model_type=resources["model_type"],
            patient_text=patient_text,
            patient_hpo_terms=list(patient_hpo_terms),
            hpo_labels=self.hpo_labels,
            ic_values=self.ic_values,
            top_k=top_k,
            result_method_name=model_name,
            patient=self.patient,
        )

    def run_stats(
        self,
        model_name: str,
        rankings: list[SimilarityResult],
        elapsed: float,
    ) -> RunStats:
        """Build run statistics for the most recent rank() call."""
        patient_terms = self.patient.get_terms(use_propagated=False)
        labels_used = select_phenotype_labels(list(patient_terms), self.hpo_labels)

        resources = self.model_registry.get(model_name)
        n_diseases_scored = (
            len(resources["disease_ids"]) if resources else len(rankings)
        )

        return build_run_stats(
            n_patient_terms_raw=len(patient_terms),
            n_patient_terms_propagated=0,
            n_patient_terms_used=len(labels_used),
            n_diseases_scored=n_diseases_scored,
            n_diseases_skipped=0,
            computation_time=elapsed,
        )
