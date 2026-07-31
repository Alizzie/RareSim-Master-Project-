"""Unit tests for semantic"""

import pytest
from raresim.similarity_methods.semantic.methods import (
    get_common_ancestors,
    get_mica,
    resnik_similarity,
    lin_similarity,
    jiang_conrath_similarity,
    best_match_scores,
    clear_mica_cache,
)


# Fixtures

@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear MICA cache before each test to avoid state leakage."""
    clear_mica_cache()


def _ancestor_sets() -> dict[str, set[str]]:
    """
    Minimal HPO-like hierarchy:
      ROOT <- A <- B
      ROOT <- A <- C
    """
    return {
        "ROOT": {"ROOT"},
        "A":    {"ROOT", "A"},
        "B":    {"ROOT", "A", "B"},
        "C":    {"ROOT", "A", "C"},
    }


def _ic_values() -> dict[str, float]:
    return {
        "ROOT": 0.0,
        "A":    1.0,
        "B":    3.0,
        "C":    2.5,
    }


# Common ancestors

def test_common_ancestors_siblings() -> None:
    """Siblings B and C share ROOT and A as common ancestors."""
    common = get_common_ancestors("B", "C", _ancestor_sets())
    assert common == {"ROOT", "A"}


def test_common_ancestors_same_term() -> None:
    """A term shares all its ancestors with itself."""
    common = get_common_ancestors("B", "B", _ancestor_sets())
    assert "B" in common and "A" in common and "ROOT" in common


def test_common_ancestors_no_overlap() -> None:
    """Terms with no shared ancestors return empty set."""
    ancestor_sets = {"X": {"X"}, "Y": {"Y"}}
    common = get_common_ancestors("X", "Y", ancestor_sets)
    assert common == set()


# MICA 

def test_mica_siblings_returns_a() -> None:
    """MICA of B and C should be A (highest IC among common ancestors ROOT, A)."""
    mica_term, mica_ic = get_mica("B", "C", _ancestor_sets(), _ic_values())
    assert mica_term == "A"
    assert mica_ic == pytest.approx(1.0)


def test_mica_is_symmetric() -> None:
    """MICA should be symmetric: (B, C) == (C, B)."""
    result_bc = get_mica("B", "C", _ancestor_sets(), _ic_values())
    clear_mica_cache()
    result_cb = get_mica("C", "B", _ancestor_sets(), _ic_values())
    assert result_bc == result_cb


def test_mica_no_common_returns_none() -> None:
    """Terms with no common ancestors should return (None, 0.0)."""
    ancestor_sets = {"X": {"X"}, "Y": {"Y"}}
    mica_term, mica_ic = get_mica("X", "Y", ancestor_sets, {})
    assert mica_term is None
    assert mica_ic == pytest.approx(0.0)


# Resnik

def test_resnik_returns_mica_ic() -> None:
    """Resnik similarity should equal IC of MICA."""
    score, mica = resnik_similarity("B", "C", _ancestor_sets(), _ic_values())
    assert score == pytest.approx(1.0)
    assert mica == "A"


def test_resnik_identical_terms() -> None:
    """Resnik of a term with itself should equal its own IC."""
    score, _ = resnik_similarity("B", "B", _ancestor_sets(), _ic_values())
    assert score == pytest.approx(3.0)


def test_resnik_no_common_ancestor() -> None:
    """Resnik with no common ancestor should return 0.0."""
    ancestor_sets = {"X": {"X"}, "Y": {"Y"}}
    score, mica = resnik_similarity("X", "Y", ancestor_sets, {})
    assert score == pytest.approx(0.0)
    assert mica is None


# Lin 

def test_lin_bounded_between_zero_and_one() -> None:
    """Lin similarity should be in [0, 1]."""
    score, _ = lin_similarity("B", "C", _ancestor_sets(), _ic_values())
    assert 0.0 <= score <= 1.0


def test_lin_identical_terms_is_one() -> None:
    """Lin similarity of a term with itself should be 1.0."""
    score, _ = lin_similarity("B", "B", _ancestor_sets(), _ic_values())
    assert score == pytest.approx(1.0)


def test_lin_no_common_ancestor_is_zero() -> None:
    """Lin similarity with no common ancestor should return 0.0."""
    ancestor_sets = {"X": {"X"}, "Y": {"Y"}}
    score, _ = lin_similarity("X", "Y", ancestor_sets, {})
    assert score == pytest.approx(0.0)


def test_lin_formula() -> None:
    """Lin = 2*IC(MICA) / (IC(a) + IC(b)) = 2*1 / (3 + 2.5)."""
    score, _ = lin_similarity("B", "C", _ancestor_sets(), _ic_values())
    assert score == pytest.approx(2.0 / (3.0 + 2.5))


# Jiang-Conrath 

def test_jc_identical_terms() -> None:
    """JC similarity of a term with itself should be 1.0 (distance = 0)."""
    score, _ = jiang_conrath_similarity("B", "B", _ancestor_sets(), _ic_values())
    assert score == pytest.approx(1.0)


def test_jc_no_common_ancestor_is_zero() -> None:
    """JC similarity with no common ancestor should return 0.0."""
    ancestor_sets = {"X": {"X"}, "Y": {"Y"}}
    score, _ = jiang_conrath_similarity("X", "Y", ancestor_sets, {})
    assert score == pytest.approx(0.0)


def test_jc_bounded() -> None:
    """JC similarity should be in (0, 1]."""
    score, _ = jiang_conrath_similarity("B", "C", _ancestor_sets(), _ic_values())
    assert 0.0 < score <= 1.0


# BMA 

def test_bma_identical_sets() -> None:
    """BMA of identical single-term sets should equal self-similarity."""
    score, details = best_match_scores(
        {"B"}, {"B"}, _ancestor_sets(), _ic_values(), resnik_similarity
    )
    assert score == pytest.approx(3.0)
    assert len(details) == 1


def test_bma_empty_source_returns_zero() -> None:
    """BMA with empty source set should return 0.0."""
    score, details = best_match_scores(
        set(), {"B"}, _ancestor_sets(), _ic_values(), resnik_similarity
    )
    assert score == pytest.approx(0.0)
    assert details == []


def test_bma_averages_best_matches() -> None:
    """BMA over two source terms should average their best match scores."""
    score, details = best_match_scores(
        {"B", "C"}, {"B", "C"}, _ancestor_sets(), _ic_values(), resnik_similarity
    )
    assert score > 0.0
    assert len(details) == 2


def test_bma_ranking_order() -> None:
    """BMA score when source is a good match should exceed a poor match."""
    good_score, _ = best_match_scores(
        {"B"}, {"B"}, _ancestor_sets(), _ic_values(), resnik_similarity
    )
    poor_score, _ = best_match_scores(
        {"B"}, {"C"}, _ancestor_sets(), _ic_values(), resnik_similarity
    )
    assert good_score > poor_score
