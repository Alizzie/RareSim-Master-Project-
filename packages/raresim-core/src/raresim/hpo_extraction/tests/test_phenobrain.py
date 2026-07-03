"""Tests for PhenoBrain API HPO extraction."""

# pylint: disable=protected-access

from pytest import MonkeyPatch

from raresim.hpo_extraction import phenobrain
from raresim.hpo_extraction._types import ExtractionMethod


def test_extract_phenobrain_api_returns_empty_when_submit_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api returns an empty list when submission fails."""
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: None)

    results = phenobrain.extract_phenobrain_api(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results


def test_extract_phenobrain_api_returns_empty_when_poll_fails(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api returns an empty list when polling fails."""
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: "task-1")
    monkeypatch.setattr(phenobrain, "_poll_result", lambda task_id: None)

    results = phenobrain.extract_phenobrain_api(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results


def test_extract_phenobrain_api_converts_mocked_result(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api correctly converts a mocked result."""
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: "task-1")
    monkeypatch.setattr(
        phenobrain,
        "_poll_result",
        lambda task_id: (
            ["HP:0001250"],
            {"HP:0001250": {"ENG_NAME": "Seizure"}},
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = phenobrain.extract_phenobrain_api("Patient has seizure.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].label == "Seizure"
    assert results[0].matched_text == "HP:0001250"
    assert results[0].method == ExtractionMethod.PHENOBRAIN_API
    assert results[0].confidence == 0.85


def test_extract_phenobrain_api_deduplicates_hpo_ids(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api deduplicates HPO IDs in the result."""
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: "task-1")
    monkeypatch.setattr(
        phenobrain,
        "_poll_result",
        lambda task_id: (
            ["HP:0001250", "HP:0001250"],
            {"HP:0001250": {"ENG_NAME": "Seizure"}},
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = phenobrain.extract_phenobrain_api("Patient has seizure.", hpo_labels)

    assert len(results) == 1


def test_extract_phenobrain_api_skips_blocklisted_terms(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api skips blocklisted HPO IDs in the result."""
    monkeypatch.setattr(phenobrain, "HPO_BLOCKLIST", {"HP:9999999"})
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: "task-1")
    monkeypatch.setattr(
        phenobrain,
        "_poll_result",
        lambda task_id: (
            ["HP:9999999", "HP:0001250"],
            {
                "HP:9999999": {"ENG_NAME": "Blocked term"},
                "HP:0001250": {"ENG_NAME": "Seizure"},
            },
        ),
    )

    hpo_labels = {
        "HP:9999999": "Blocked term",
        "HP:0001250": "Seizure",
    }

    results = phenobrain.extract_phenobrain_api("Patient has seizure.", hpo_labels)

    assert [result.hpo_id for result in results] == ["HP:0001250"]


def test_extract_phenobrain_api_falls_back_to_hpo_labels(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extract_phenobrain_api falls back to HPO labels when API label is missing."""
    monkeypatch.setattr(phenobrain, "_submit_text", lambda raw_text: "task-1")
    monkeypatch.setattr(
        phenobrain,
        "_poll_result",
        lambda task_id: (
            ["HP:0001250"],
            {"HP:0001250": {}},
        ),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = phenobrain.extract_phenobrain_api("Patient has seizure.", hpo_labels)

    assert len(results) == 1
    assert results[0].label == "Seizure"


def test_parse_hpo_list_ignores_invalid_values() -> None:
    """Test that _parse_hpo_list ignores invalid values."""
    parsed = phenobrain._parse_hpo_list(["HP:0001250", None, "", 123])

    assert parsed == ["HP:0001250"]


def test_parse_hpo_to_info_ignores_invalid_values() -> None:
    """Test that _parse_hpo_to_info ignores invalid values."""
    parsed = phenobrain._parse_hpo_to_info(
        {
            "HP:0001250": {"ENG_NAME": "Seizure"},
            "HP:0001251": "invalid",
            123: {"ENG_NAME": "Invalid"},
        }
    )

    assert parsed == {"HP:0001250": {"ENG_NAME": "Seizure"}}
