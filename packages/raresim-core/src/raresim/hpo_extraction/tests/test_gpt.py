"""Tests for GPT-based HPO extraction."""

# pylint: disable=protected-access,too-few-public-methods

from collections.abc import Callable
from typing import Any

from pytest import MonkeyPatch

from raresim.hpo_extraction import gpt
from raresim.hpo_extraction._types import ExtractionMethod


def _mock_request_phenotypes(
    phenotypes: list[str],
) -> Callable[[str, str, bool], list[str]]:
    """Return a fake _request_phenotypes function."""

    def request_phenotypes(
        _raw_text: str,
        _model: str,
        _skip_negated: bool,
    ) -> list[str]:
        """Return predefined phenotype phrases."""
        return phenotypes

    return request_phenotypes


def test_extract_chatgpt_maps_mocked_phrases(monkeypatch: MonkeyPatch) -> None:
    """Test that mocked GPT phrases are mapped to HPO IDs."""
    monkeypatch.setattr(
        gpt,
        "_request_phenotypes",
        _mock_request_phenotypes(["seizure", "unknown phrase"]),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = gpt.extract_chatgpt("Patient has seizures.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].label == "Seizure"
    assert results[0].matched_text == "seizure"
    assert results[0].method == ExtractionMethod.CHATGPT
    assert results[0].confidence == 0.85


def test_extract_chatgpt_removes_duplicate_hpo_ids(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that duplicate GPT phrases produce only one HPO result."""
    monkeypatch.setattr(
        gpt,
        "_request_phenotypes",
        _mock_request_phenotypes(["seizure", "seizure"]),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = gpt.extract_chatgpt("Patient has seizure.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"


def test_extract_chatgpt_skips_blocklisted_terms(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that GPT extraction skips blocklisted HPO terms."""
    monkeypatch.setattr(
        gpt,
        "_request_phenotypes",
        _mock_request_phenotypes(["all"]),
    )
    monkeypatch.setattr(gpt, "HPO_BLOCKLIST", {"HP:0000001"})

    hpo_labels = {"HP:0000001": "All"}

    results = gpt.extract_chatgpt("Patient text.", hpo_labels)

    assert not results


def test_extract_chatgpt_uses_conservative_fallback_match(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that GPT extraction uses conservative containment matching."""
    monkeypatch.setattr(
        gpt,
        "_request_phenotypes",
        _mock_request_phenotypes(["developmental delay"]),
    )

    hpo_labels = {"HP:0001263": "Global developmental delay"}

    results = gpt.extract_chatgpt("Patient has developmental delay.", hpo_labels)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001263"


def test_extract_chatgpt_ignores_unknown_phrases(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that unmapped GPT phrases are ignored."""
    monkeypatch.setattr(
        gpt,
        "_request_phenotypes",
        _mock_request_phenotypes(["unknown phrase"]),
    )

    hpo_labels = {"HP:0001250": "Seizure"}

    results = gpt.extract_chatgpt("Patient text.", hpo_labels)

    assert not results


def test_parse_phenotypes_handles_json_object() -> None:
    """Test parsing phenotype phrases from a valid JSON object."""
    content = '{"phenotypes": ["seizure", "hypotonia"]}'

    phenotypes = gpt._parse_phenotypes(content)

    assert phenotypes == ["seizure", "hypotonia"]


def test_parse_phenotypes_strips_markdown_fence() -> None:
    """Test parsing phenotype phrases from fenced JSON."""
    content = '```json\n{"phenotypes": ["seizure"]}\n```'

    phenotypes = gpt._parse_phenotypes(content)

    assert phenotypes == ["seizure"]


def test_parse_phenotypes_returns_empty_for_non_list() -> None:
    """Test that non-list phenotype JSON returns an empty list."""
    content = '{"phenotypes": "seizure"}'

    phenotypes = gpt._parse_phenotypes(content)

    assert not phenotypes


def test_request_phenotypes_returns_empty_for_invalid_json(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that invalid GPT JSON output is handled safely."""

    class FakeMessage:
        """Fake OpenAI message."""

        content = "not valid json"

    class FakeChoice:
        """Fake OpenAI response choice."""

        message = FakeMessage()

    class FakeResponse:
        """Fake OpenAI response."""

        choices = [FakeChoice()]

    class FakeCompletions:
        """Fake OpenAI completions API."""

        @staticmethod
        def create(**_kwargs: Any) -> FakeResponse:
            """Return a fake response with invalid JSON."""
            return FakeResponse()

    class FakeChat:
        """Fake OpenAI chat API."""

        completions = FakeCompletions()

    class FakeClient:
        """Fake OpenAI client."""

        chat = FakeChat()

    def get_fake_openai_client() -> FakeClient:
        """Return a fake OpenAI client."""
        return FakeClient()

    monkeypatch.setattr(gpt, "_get_openai_client", get_fake_openai_client)

    phenotypes = gpt._request_phenotypes(
        raw_text="Patient has seizure.",
        model="fake-model",
        skip_negated=True,
    )

    assert not phenotypes
