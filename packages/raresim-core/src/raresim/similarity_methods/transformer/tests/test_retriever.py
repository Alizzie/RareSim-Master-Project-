"""Tests for transformer-based disease retrieval."""

# pylint: disable=protected-access

from types import MethodType
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

from raresim.similarity_methods.transformer import retriever as transformer_retriever
from raresim.types.result import SimilarityResult
from raresim.types.schemas import PatientProfile


MODEL_NAME = "fake-transformer-model"


def _patient() -> PatientProfile:
    """Build a small patient profile for transformer tests."""
    return PatientProfile(
        patient_id="patient-1",
        raw_text="Patient has seizure.",
        hpo_terms={"HP:0001250"},
        propagated_hpo_terms={"HP:0001250", "HP:0000118"},
    )


def _disease_profiles() -> dict[str, dict[str, Any]]:
    """Build tiny fake disease profiles."""
    return {
        "ORPHA:1": {
            "label": "Disease A",
            "hpo_terms": ["HP:0001250"],
            "propagated_hpo_terms": ["HP:0001250", "HP:0000118"],
        },
        "ORPHA:2": {
            "label": "Disease B",
            "hpo_terms": ["HP:0004322"],
            "propagated_hpo_terms": ["HP:0004322", "HP:0000118"],
        },
        "ORPHA:3": {
            "label": "Disease C",
            "hpo_terms": ["HP:0001250", "HP:0004322"],
            "propagated_hpo_terms": ["HP:0001250", "HP:0004322", "HP:0000118"],
        },
    }


def _patch_common_dependencies(monkeypatch: MonkeyPatch) -> None:
    """Patch expensive or external transformer dependencies."""

    def fake_build_disease_texts(
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return deterministic disease texts."""
        return (
            ["ORPHA:1", "ORPHA:2", "ORPHA:3"],
            ["Disease A", "Disease B", "Disease C"],
            ["text disease a", "text disease b", "text disease c"],
        )

    def fake_build_patient_text(
        *_args: Any,
        **_kwargs: Any,
    ) -> str:
        """Return deterministic patient text."""
        return "patient text"


    def fake_category_metadata(**_kwargs: Any) -> dict[str, Any]:
        """Return minimal category metadata."""
        return {
            "profile_type": "canonical",
            "category_source_id": None,
            "category_path": [],
            "matched_aliases": [],
        }

    def fake_method_specific_block(**kwargs: Any) -> dict[str, Any]:
        """Return a minimal method-specific explanation block."""
        return {
            "model_name": kwargs.get("model_name"),
            "method_name": kwargs.get("method_name"),
        }

    def fake_explanation(**kwargs: Any) -> dict[str, Any]:
        """Return a minimal explanation dictionary."""
        return {
            "summary": "fake transformer explanation",
            "method_specific": kwargs.get("method_specific", {}),
            "diagnostics": {"raw_score": kwargs.get("score")},
        }

    monkeypatch.setattr(
        transformer_retriever,
        "build_disease_texts",
        fake_build_disease_texts,
    )
    monkeypatch.setattr(
        transformer_retriever,
        "build_patient_text",
        fake_build_patient_text,
    )
    monkeypatch.setattr(
        transformer_retriever,
        "build_category_metadata",
        fake_category_metadata,
    )
    monkeypatch.setattr(
        transformer_retriever,
        "build_method_specific_explanation_block",
        fake_method_specific_block,
    )
    monkeypatch.setattr(transformer_retriever, "build_explanation", fake_explanation)


def _make_retriever(monkeypatch: MonkeyPatch) -> transformer_retriever.DiseaseRetriever:
    """Create a transformer DiseaseRetriever with patched dependencies."""
    _patch_common_dependencies(monkeypatch)

    return transformer_retriever.DiseaseRetriever(
        disease_profiles=_disease_profiles(),
        hpo_labels={
            "HP:0001250": "Seizure",
            "HP:0004322": "Short stature",
            "HP:0000118": "Phenotypic abnormality",
        },
        alias_to_canonical={},
        model_list=[MODEL_NAME],
        patient=_patient(),
        disease_ancestors={},
        disease_metadata_index={},
        ic_values={},
    )


def _install_fake_model_registry(
    retriever: transformer_retriever.DiseaseRetriever,
) -> None:
    """Install fake disease embeddings directly into the model registry."""
    resources: dict[str, Any] = {
        "model_type": "sentence_transformer",
        "disease_ids": ["ORPHA:1", "ORPHA:2", "ORPHA:3"],
        "disease_labels": ["Disease A", "Disease B", "Disease C"],
        "disease_texts": ["text disease a", "text disease b", "text disease c"],
        "disease_embeddings": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        ),
        "cache_metadata": {"pooling": None},
    }

    retriever.model_registry[MODEL_NAME] = resources

    text_mode = getattr(retriever, "text_mode", None)
    if text_mode is not None:
        retriever.model_registry[f"{MODEL_NAME}::{text_mode}"] = resources


def test_rank_orders_diseases_by_embedding_similarity(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that rank orders diseases by embedding similarity."""
    retriever = _make_retriever(monkeypatch)
    _install_fake_model_registry(retriever)

    def skip_resource_loading(
        _self: transformer_retriever.DiseaseRetriever,
        _model_name: str,
        backend: dict[str, Any] | None = None,
    ) -> None:
        """Skip persistent cache and model loading."""
        _ = backend

    def fake_patient_embedding(
        _self: transformer_retriever.DiseaseRetriever,
        _model_name: str,
        _patient_text: str,
    ) -> np.ndarray:
        """Return a deterministic patient embedding."""
        return np.array([1.0, 0.0])

    monkeypatch.setattr(
        retriever,
        "_ensure_model_resources",
        MethodType(skip_resource_loading, retriever),
    )
    monkeypatch.setattr(
        retriever,
        "_get_patient_embedding",
        MethodType(fake_patient_embedding, retriever),
    )

    results = retriever.rank(
        model_name=MODEL_NAME,
        patient=_patient(),
        top_k=2,
        candidate_pool_size=3,
    )

    assert [result.disease_id for result in results] == ["ORPHA:1", "ORPHA:3"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score
    assert MODEL_NAME in results[0].method_name


def test_rank_rejects_unknown_model(monkeypatch: MonkeyPatch) -> None:
    """Test that rank rejects models outside the configured model list."""
    retriever = _make_retriever(monkeypatch)

    with pytest.raises(ValueError, match="Model not available"):
        retriever.rank(
            model_name="unknown-model",
            patient=_patient(),
            top_k=2,
        )


def test_get_patient_embedding_uses_in_memory_cache(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that patient embeddings are cached for repeated patient text."""
    retriever = _make_retriever(monkeypatch)
    calls: list[list[str]] = []

    def fake_hash_text(_text: str) -> str:
        """Return a stable fake text hash."""
        return "same-hash"

    def fake_get_backend(
        _self: transformer_retriever.DiseaseRetriever,
        _model_name: str,
    ) -> dict[str, str]:
        """Return a fake embedding backend."""
        return {"backend": "fake"}

    def fake_embed_texts(
        _backend: dict[str, str],
        texts: list[str],
    ) -> np.ndarray:
        """Return one fake embedding and record calls."""
        calls.append(texts)
        return np.array([[0.25, 0.75]])

    monkeypatch.setattr(transformer_retriever, "hash_text", fake_hash_text)
    monkeypatch.setattr(transformer_retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        retriever,
        "_get_backend",
        MethodType(fake_get_backend, retriever),
    )

    first = retriever._get_patient_embedding(MODEL_NAME, "same patient text")
    second = retriever._get_patient_embedding(MODEL_NAME, "same patient text")

    assert len(calls) == 1
    assert np.array_equal(first, second)


def test_run_stats_counts_patient_and_disease_terms(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test transformer run statistics from retriever model resources."""
    retriever = _make_retriever(monkeypatch)
    _install_fake_model_registry(retriever)

    rankings = [
        SimilarityResult(
            disease_id="ORPHA:1",
            label="Disease A",
            score=1.0,
            method_name=MODEL_NAME,
            rank=1,
        ),
        SimilarityResult(
            disease_id="ORPHA:2",
            label="Disease B",
            score=0.5,
            method_name=MODEL_NAME,
            rank=2,
        ),
    ]

    stats = retriever.run_stats(
        model_name=MODEL_NAME,
        rankings=rankings,
        elapsed=1.25,
    )

    assert stats.n_patient_terms_raw == 1
    assert stats.n_patient_terms_propagated == 0
    assert stats.n_patient_terms_used == 1
    assert stats.n_diseases_scored == 3
    assert stats.n_diseases_skipped == 0
    assert stats.computation_time_seconds == 1.25
