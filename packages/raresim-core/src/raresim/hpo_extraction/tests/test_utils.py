"""Tests for shared HPO extraction utilities."""

from raresim.hpo_extraction import _utils as utils
from raresim.hpo_extraction._types import ExtractionMethod, ExtractionResult


def _result(
    hpo_id: str,
    confidence: float,
    start: int | None = None,
) -> ExtractionResult:
    """Build a minimal extraction result for tests."""
    return ExtractionResult(
        hpo_id=hpo_id,
        label=hpo_id,
        matched_text=hpo_id,
        method=ExtractionMethod.DICTIONARY,
        confidence=confidence,
        start=start,
        end=None,
        negated=False,
    )


def test_normalize_text_lowercases_and_removes_punctuation() -> None:
    """Test that normalize_text lowercases and removes punctuation."""
    text = "Patient has Seizures, Ataxia!!!"
    assert utils.normalize_text(text) == "patient has seizures ataxia"


def test_normalize_text_keeps_hyphen() -> None:
    """Test that normalize_text keeps hyphens in words."""
    text = "Short-stature."
    assert utils.normalize_text(text) == "short-stature"


def test_is_negated_detects_negation_before_mention() -> None:
    """Test that is_negated detects negation words before a mention."""
    text = utils.normalize_text("The patient has no seizure.")
    start = text.index("seizure")

    assert utils.is_negated(text, start) is True


def test_is_negated_returns_false_without_negation() -> None:
    """Test that is_negated returns False when no negation is present."""
    text = utils.normalize_text("The patient has seizure.")
    start = text.index("seizure")

    assert utils.is_negated(text, start) is False


def test_build_label_lookup_normalizes_labels() -> None:
    """Test that build_label_lookup normalizes HPO labels."""
    hpo_labels = {
        "HP:0001250": "Seizure",
        "HP:0001252": "Muscular hypotonia",
    }

    lookup = utils.build_label_lookup(hpo_labels)

    assert lookup["seizure"] == "HP:0001250"
    assert lookup["muscular hypotonia"] == "HP:0001252"


def test_deduplicate_keeps_highest_confidence() -> None:
    """Test that deduplicate keeps the result with the highest confidence."""
    results = [
        _result("HP:0001250", confidence=0.4),
        _result("HP:0001250", confidence=0.9),
    ]

    deduplicated = utils.deduplicate(results)

    assert len(deduplicated) == 1
    assert deduplicated[0].hpo_id == "HP:0001250"
    assert deduplicated[0].confidence == 0.9


def test_deduplicate_sorts_by_start_then_hpo_id() -> None:
    """Test that deduplicate sorts results by start offset, then HPO ID."""
    results = [
        _result("HP:9999002", confidence=1.0, start=10),
        _result("HP:9999001", confidence=1.0, start=5),
    ]

    deduplicated = utils.deduplicate(results)

    assert [result.hpo_id for result in deduplicated] == [
        "HP:9999001",
        "HP:9999002",
    ]


def test_deduplicate_skips_blocklisted_terms(monkeypatch) -> None:
    """Test that deduplicate skips results with blocklisted HPO IDs."""
    monkeypatch.setattr(utils, "HPO_BLOCKLIST", {"HP:9999999"})

    results = [
        _result("HP:9999999", confidence=1.0),
        _result("HP:0001250", confidence=1.0),
    ]

    deduplicated = utils.deduplicate(results)

    assert [result.hpo_id for result in deduplicated] == ["HP:0001250"]
