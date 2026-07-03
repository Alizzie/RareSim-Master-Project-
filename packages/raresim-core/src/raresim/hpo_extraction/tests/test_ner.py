"""Tests for biomedical NER HPO extraction."""

# pylint: disable=protected-access

from collections.abc import Callable
from typing import Any

from pytest import MonkeyPatch

from raresim.hpo_extraction import ner
from raresim.hpo_extraction._types import ExtractionMethod


def _mock_pipeline(
    entities: list[dict[str, Any]],
) -> Callable[[str], list[dict[str, Any]]]:
    """Return a fake HuggingFace pipeline."""

    def run_pipeline(_raw_text: str) -> list[dict[str, Any]]:
        return entities

    return run_pipeline


def test_extract_biomedical_ner_maps_entity(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner correctly maps a mocked entity to HPO ID."""
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": "seizure",
                    "score": 0.99,
                    "start": 16,
                    "end": 23,
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = ner.extract_biomedical_ner("Patient has a seizure.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].method == ExtractionMethod.BIOMEDICAL_NER
    assert results[0].confidence == 0.99
    assert results[0].negated is False


def test_extract_biomedical_ner_skips_low_confidence(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner skips entities below the confidence threshold."""
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.8)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": "seizure",
                    "score": 0.2,
                    "start": 16,
                    "end": 23,
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = ner.extract_biomedical_ner("Patient has a seizure.", hpo_labels)

    assert not results


def test_extract_biomedical_ner_skips_negated_entity(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner skips negated entities by default."""
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": "seizure",
                    "score": 0.99,
                    "start": 15,
                    "end": 22,
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = ner.extract_biomedical_ner("Patient has no seizure.", hpo_labels)

    assert not results


def test_extract_biomedical_ner_keeps_negated_entity_when_requested(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner keeps negated entities when skip_negated=False."""
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": "seizure",
                    "score": 0.99,
                    "start": 15,
                    "end": 22,
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = ner.extract_biomedical_ner(
        "Patient has no seizure.",
        hpo_labels,
        skip_negated=False,
    )

    assert len(results) == 1
    assert results[0].negated is True


def test_extract_biomedical_ner_uses_fallback_containment_match(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner uses fallback containment match when entity is a substring."""  # pylint: disable=line-too-long
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": "developmental delay",
                    "score": 0.99,
                    "start": 12,
                    "end": 31,
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001263": "Global developmental delay"}

    results = ner.extract_biomedical_ner(
        "Patient has developmental delay.",
        hpo_labels,
    )

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001263"


def test_extract_biomedical_ner_ignores_bad_entity_shape(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_biomedical_ner ignores entities with bad shape."""
    monkeypatch.setattr(ner, "BIOMEDICAL_NER_MIN_CONFIDENCE", 0.5)
    monkeypatch.setattr(
        ner,
        "_get_ner_pipeline",
        lambda model_name: _mock_pipeline(
            [
                {
                    "word": None,
                    "score": "bad-score",
                }
            ]
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = ner.extract_biomedical_ner("Patient has seizure.", hpo_labels)

    assert not results
