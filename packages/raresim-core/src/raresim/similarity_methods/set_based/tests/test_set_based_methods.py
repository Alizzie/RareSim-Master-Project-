"""Unit tests for set-based"""

import pytest
from raresim.similarity_methods.set_based.methods import (
    jaccard_similarity,
    dice_similarity,
    overlap_coefficient,
    cosine_similarity,
)


# Fixtures

def _terms_a() -> set[str]:
    return {"HP:0001250", "HP:0000545", "HP:0001263"}


def _terms_b() -> set[str]:
    return {"HP:0001250", "HP:0000545", "HP:0004322"}


def _disjoint() -> set[str]:
    return {"HP:0009999"}


# Jaccard

def test_jaccard_identical_sets() -> None:
    """Jaccard of identical sets should be 1.0."""
    terms = _terms_a()
    assert jaccard_similarity(terms, terms) == pytest.approx(1.0)


def test_jaccard_partial_overlap() -> None:
    """Jaccard with 2/4 union terms should be 0.5."""
    a = _terms_a()
    b = _terms_b()
    # intersection = {HP:0001250, HP:0000545}, union = 4 terms
    assert jaccard_similarity(a, b) == pytest.approx(2 / 4)


def test_jaccard_disjoint_sets() -> None:
    """Jaccard of disjoint sets should be 0.0."""
    assert jaccard_similarity(_terms_a(), _disjoint()) == pytest.approx(0.0)


def test_jaccard_empty_input() -> None:
    """Jaccard with an empty set should return 0.0."""
    assert jaccard_similarity(set(), _terms_a()) == pytest.approx(0.0)
    assert jaccard_similarity(_terms_a(), set()) == pytest.approx(0.0)


# Dice 

def test_dice_identical_sets() -> None:
    """Dice of identical sets should be 1.0."""
    terms = _terms_a()
    assert dice_similarity(terms, terms) == pytest.approx(1.0)


def test_dice_partial_overlap() -> None:
    """Dice with 2 shared out of 3+3 terms should be 2*2/(3+3)."""
    a = _terms_a()
    b = _terms_b()
    assert dice_similarity(a, b) == pytest.approx(2 * 2 / (3 + 3))


def test_dice_disjoint_sets() -> None:
    """Dice of disjoint sets should be 0.0."""
    assert dice_similarity(_terms_a(), _disjoint()) == pytest.approx(0.0)


def test_dice_empty_input() -> None:
    """Dice with an empty set should return 0.0."""
    assert dice_similarity(set(), _terms_a()) == pytest.approx(0.0)


# ── Overlap ───────────────────────────────────────────────────────────────────

def test_overlap_identical_sets() -> None:
    """Overlap coefficient of identical sets should be 1.0."""
    terms = _terms_a()
    assert overlap_coefficient(terms, terms) == pytest.approx(1.0)


def test_overlap_subset() -> None:
    """Overlap coefficient when one set is a subset of the other should be 1.0."""
    large = _terms_a()
    small = {"HP:0001250", "HP:0000545"}
    assert overlap_coefficient(small, large) == pytest.approx(1.0)


def test_overlap_disjoint_sets() -> None:
    """Overlap coefficient of disjoint sets should be 0.0."""
    assert overlap_coefficient(_terms_a(), _disjoint()) == pytest.approx(0.0)


def test_overlap_empty_input() -> None:
    """Overlap with an empty set should return 0.0."""
    assert overlap_coefficient(set(), _terms_a()) == pytest.approx(0.0)


# ── Cosine ────────────────────────────────────────────────────────────────────

def test_cosine_identical_sets() -> None:
    """Cosine similarity of identical sets should be 1.0."""
    terms = _terms_a()
    assert cosine_similarity(terms, terms) == pytest.approx(1.0)


def test_cosine_disjoint_sets() -> None:
    """Cosine similarity of disjoint sets should be 0.0."""
    assert cosine_similarity(_terms_a(), _disjoint()) == pytest.approx(0.0)


def test_cosine_empty_input() -> None:
    """Cosine similarity with an empty set should return 0.0."""
    assert cosine_similarity(set(), _terms_a()) == pytest.approx(0.0)


def test_cosine_partial_overlap_bounded() -> None:
    """Cosine similarity with partial overlap should be between 0 and 1."""
    score = cosine_similarity(_terms_a(), _terms_b())
    assert 0.0 < score < 1.0
