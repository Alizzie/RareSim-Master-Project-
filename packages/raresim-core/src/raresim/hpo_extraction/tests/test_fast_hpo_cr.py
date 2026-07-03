"""Tests for FastHPOCR HPO extraction."""

# pylint: disable=protected-access,too-few-public-methods

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from pytest import MonkeyPatch

from raresim.hpo_extraction import fast_hpo_cr
from raresim.hpo_extraction._types import ExtractionMethod


def _annotation(
    hpo_uri: str | None,
    text_span: str,
    start_offset: int | None,
    end_offset: int | None,
) -> SimpleNamespace:
    """Build a fake FastHPOCR annotation object."""
    return SimpleNamespace(
        hpoUri=hpo_uri,
        textSpan=text_span,
        startOffset=start_offset,
        endOffset=end_offset,
    )


def _getter_for(annotator: Any | None) -> Callable[[], Any | None]:
    """Return a fake _get_fast_hpo_cr function."""

    def get_annotator() -> Any | None:
        """Return the predefined fake annotator."""
        return annotator

    return get_annotator


class FakeAnnotator:
    """Fake FastHPOCR annotator."""

    def __init__(self, annotations: list[SimpleNamespace]) -> None:
        """Store fake annotations."""
        self._annotations = annotations

    def annotate(self, _text: str) -> list[SimpleNamespace]:
        """Return fake annotations."""
        return self._annotations


class FailingAnnotator:
    """Fake FastHPOCR annotator that raises during annotation."""

    @staticmethod
    def annotate(_text: str) -> list[SimpleNamespace]:
        """Raise an error to simulate a FastHPOCR failure."""
        raise RuntimeError("annotation failed")


def test_extract_fast_hpo_cr_returns_empty_when_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that extraction returns an empty list when FastHPOCR is unavailable."""
    monkeypatch.setattr(fast_hpo_cr, "_get_fast_hpo_cr", _getter_for(None))

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results


def test_extract_fast_hpo_cr_converts_annotations(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that FastHPOCR annotations are converted into extraction results."""
    annotator = FakeAnnotator(
        [
            _annotation(
                hpo_uri="HP:0001250",
                text_span="seizure",
                start_offset=12,
                end_offset=19,
            )
        ]
    )
    monkeypatch.setattr(fast_hpo_cr, "_get_fast_hpo_cr", _getter_for(annotator))

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0001250"
    assert results[0].label == "Seizure"
    assert results[0].matched_text == "seizure"
    assert results[0].method == ExtractionMethod.FAST_HPO_CR
    assert results[0].confidence == 0.90
    assert results[0].start == 12
    assert results[0].end == 19
    assert results[0].negated is False


def test_extract_fast_hpo_cr_skips_annotation_without_hpo_id(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that annotations without an HPO ID are skipped."""
    annotator = FakeAnnotator(
        [
            _annotation(
                hpo_uri=None,
                text_span="seizure",
                start_offset=12,
                end_offset=19,
            )
        ]
    )
    monkeypatch.setattr(fast_hpo_cr, "_get_fast_hpo_cr", _getter_for(annotator))

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results


def test_extract_fast_hpo_cr_skips_negated_annotation(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that negated FastHPOCR annotations are skipped by default."""
    annotator = FakeAnnotator(
        [
            _annotation(
                hpo_uri="HP:0001250",
                text_span="seizure",
                start_offset=15,
                end_offset=22,
            )
        ]
    )
    monkeypatch.setattr(fast_hpo_cr, "_get_fast_hpo_cr", _getter_for(annotator))

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has no seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results


def test_extract_fast_hpo_cr_keeps_negated_annotation_when_requested(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that negated annotations are kept when skip_negated is False."""
    annotator = FakeAnnotator(
        [
            _annotation(
                hpo_uri="HP:0001250",
                text_span="seizure",
                start_offset=15,
                end_offset=22,
            )
        ]
    )
    monkeypatch.setattr(fast_hpo_cr, "_get_fast_hpo_cr", _getter_for(annotator))

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has no seizure.",
        {"HP:0001250": "Seizure"},
        skip_negated=False,
    )

    assert len(results) == 1
    assert results[0].negated is True


def test_extract_fast_hpo_cr_handles_annotation_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that FastHPOCR annotation errors do not crash extraction."""
    monkeypatch.setattr(
        fast_hpo_cr,
        "_get_fast_hpo_cr",
        _getter_for(FailingAnnotator()),
    )

    results = fast_hpo_cr.extract_fast_hpo_cr(
        "Patient has seizure.",
        {"HP:0001250": "Seizure"},
    )

    assert not results
