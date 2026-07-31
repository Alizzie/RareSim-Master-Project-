"""Unit tests for HPO2Vec"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from raresim.similarity_methods.hpo2vec.methods import (
    build_graph,
    _transition_probs,
    random_walk,
    embed_term_set,
)


# Fixtures 

def _hpo_parents() -> dict[str, list[str]]:
    return {
        "HP:0001250": ["HP:0000118"],
        "HP:0004322": ["HP:0000118"],
    }


def _disease_profiles() -> dict[str, dict]:
    return {
        "ORPHA:1": {"hpo_terms": ["HP:0001250"]},
        "ORPHA:2": {"hpo_terms": ["HP:0004322"]},
    }


def _ic_values() -> dict[str, float]:
    return {
        "HP:0001250": 3.0,
        "HP:0004322": 2.5,
        "HP:0000118": 0.4,
    }


# build_graph

def test_build_graph_adds_is_a_edges() -> None:
    """IS-A edges should be bidirectional in the graph."""
    graph = build_graph(_hpo_parents(), {})
    assert "HP:0000118" in graph["HP:0001250"]
    assert "HP:0001250" in graph["HP:0000118"]


def test_build_graph_adds_has_phenotype_edges() -> None:
    """HAS_PHENOTYPE edges between diseases and HPO terms should be bidirectional."""
    graph = build_graph({}, _disease_profiles())
    assert "HP:0001250" in graph["ORPHA:1"]
    assert "ORPHA:1" in graph["HP:0001250"]


def test_build_graph_all_nodes_present() -> None:
    """All HPO terms and disease nodes should appear in the graph."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    for node in ["HP:0001250", "HP:0004322", "HP:0000118", "ORPHA:1", "ORPHA:2"]:
        assert node in graph


def test_build_graph_empty_inputs() -> None:
    """Empty inputs should produce an empty graph."""
    graph = build_graph({}, {})
    assert graph == {}


# _transition_probs 

def test_transition_probs_sum_to_one() -> None:
    """Transition probabilities should always sum to 1.0."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    neighbours = ["HP:0000118", "ORPHA:1"]
    probs = _transition_probs(
        current="HP:0001250",
        previous=None,
        neighbours=neighbours,
        graph=graph,
        ic_values=_ic_values(),
        p=1.0,
        q=1.0,
    )
    assert sum(probs) == pytest.approx(1.0)


def test_transition_probs_return_penalty() -> None:
    """Returning to previous node should be penalized (lower probability) when p > 1."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    neighbours = ["HP:0000118", "ORPHA:1"]
    # previous = HP:0000118, so returning to it should be penalized with p=2
    probs_high_p = _transition_probs(
        current="HP:0001250",
        previous="HP:0000118",
        neighbours=neighbours,
        graph=graph,
        ic_values=_ic_values(),
        p=2.0,
        q=1.0,
    )
    probs_low_p = _transition_probs(
        current="HP:0001250",
        previous="HP:0000118",
        neighbours=neighbours,
        graph=graph,
        ic_values=_ic_values(),
        p=0.5,
        q=1.0,
    )
    # With high p, returning to HP:0000118 (index 0) should be less likely
    assert probs_high_p[0] < probs_low_p[0]


def test_transition_probs_no_previous_uniform_bias() -> None:
    """With no previous node, bias should be 1.0 for all neighbours."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    neighbours = ["HP:0000118", "ORPHA:1"]
    probs = _transition_probs(
        current="HP:0001250",
        previous=None,
        neighbours=neighbours,
        graph=graph,
        ic_values=_ic_values(),
        p=1.0,
        q=1.0,
    )
    # Bias is 1.0 for all, so weights are purely IC-based
    ic_118 = _ic_values().get("HP:0000118", 1.0)
    ic_orpha1 = _ic_values().get("ORPHA:1", 1.0)
    total = ic_118 + ic_orpha1
    assert probs[0] == pytest.approx(ic_118 / total)
    assert probs[1] == pytest.approx(ic_orpha1 / total)


# random_walk

def test_random_walk_length() -> None:
    """Walk should have exactly walk_length nodes."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    walk = random_walk("HP:0001250", graph, _ic_values(), walk_length=5, p=1.0, q=1.0)
    assert len(walk) == 5


def test_random_walk_starts_at_node() -> None:
    """Walk should start at the given node."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    walk = random_walk("HP:0001250", graph, _ic_values(), walk_length=4, p=1.0, q=1.0)
    assert walk[0] == "HP:0001250"


def test_random_walk_stays_in_graph() -> None:
    """Every node in the walk should be in the graph."""
    graph = build_graph(_hpo_parents(), _disease_profiles())
    walk = random_walk("HP:0001250", graph, _ic_values(), walk_length=6, p=1.0, q=1.0)
    for node in walk:
        assert node in graph


def test_random_walk_isolated_node_stops() -> None:
    """Walk starting from a node with no neighbours should stop immediately."""
    graph = {"ISOLATED": []}
    walk = random_walk("ISOLATED", graph, {}, walk_length=5, p=1.0, q=1.0)
    assert walk == ["ISOLATED"]


# embed_term_set 

def _fake_word2vec(terms: list[str]) -> MagicMock:
    """Build a mock Word2Vec model with 2D embeddings for given terms."""
    embeddings = {term: np.array([float(i), float(i)]) for i, term in enumerate(terms, 1)}
    wv = MagicMock()
    wv.__contains__ = lambda self, term: term in embeddings
    wv.__getitem__ = lambda self, term: embeddings[term]
    model = MagicMock()
    model.wv = wv
    return model


def test_embed_term_set_returns_array() -> None:
    """embed_term_set should return a numpy array for known terms."""
    model = _fake_word2vec(["HP:0001250", "HP:0004322"])
    result = embed_term_set({"HP:0001250"}, model, _ic_values())
    assert isinstance(result, np.ndarray)


def test_embed_term_set_unknown_terms_returns_none() -> None:
    """embed_term_set should return None if no terms are in the vocabulary."""
    model = _fake_word2vec(["HP:0001250"])
    result = embed_term_set({"HP:9999999"}, model, _ic_values())
    assert result is None


def test_embed_term_set_ic_weighted() -> None:
    """Higher IC terms should contribute more to the embedding."""
    model = _fake_word2vec(["HP:0001250", "HP:0004322"])
    ic = {"HP:0001250": 10.0, "HP:0004322": 1.0}
    result = embed_term_set({"HP:0001250", "HP:0004322"}, model, ic)
    assert result is not None
    # HP:0001250 has embedding [1,1], HP:0004322 has [2,2]
    # With IC weights 10 and 1 normalized: result should be closer to [1,1]
    assert result[0] < 1.5  # closer to HP:0001250's embedding


def test_embed_term_set_empty_terms_returns_none() -> None:
    """embed_term_set with empty term set should return None."""
    model = _fake_word2vec(["HP:0001250"])
    result = embed_term_set(set(), model, _ic_values())
    assert result is None
