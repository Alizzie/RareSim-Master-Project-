"""Tests for dictionary-based HPO extraction."""

from raresim.hpo_extraction._types import ExtractionMethod
from raresim.hpo_extraction.dictionary import extract_dictionary


def test_extract_dictionary_finds_exact_label() -> None:
    """Test that dictionary extraction finds an exact HPO label match."""
    hpo_labels = {"HP:0001250": "Seizure"}

    results = extract_dictionary("The patient has seizure.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].label == "Seizure"
    assert results[0].matched_text == "seizure"
    assert results[0].method == ExtractionMethod.DICTIONARY
    assert results[0].confidence == 1.0
    assert results[0].negated is False


def test_extract_dictionary_skips_negated_label_by_default() -> None:
    """Test that dictionary extraction skips negated labels by default."""
    hpo_labels = {"HP:0001250": "Seizure"}

    results = extract_dictionary("The patient has no seizure.", hpo_labels)

    assert not results


def test_extract_dictionary_keeps_negated_label_when_requested() -> None:
    """Test that dictionary extraction keeps negated labels when requested."""
    hpo_labels = {"HP:0001250": "Seizure"}

    results = extract_dictionary(
        "The patient has no seizure.",
        hpo_labels,
        skip_negated=False,
    )

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].negated is True


def test_extract_dictionary_returns_empty_list_when_no_match() -> None:
    """Test that dictionary extraction returns no results when labels do not match."""
    hpo_labels = {"HP:0001250": "Seizure"}

    results = extract_dictionary("The patient has normal development.", hpo_labels)

    assert not results


def test_extract_dictionary_matches_multiword_label() -> None:
    """Test that dictionary extraction matches multiword HPO labels."""
    hpo_labels = {"HP:0001263": "Global developmental delay"}

    results = extract_dictionary(
        "The patient has global developmental delay.",
        hpo_labels,
    )

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001263"
