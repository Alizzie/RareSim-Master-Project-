"""Unit tests for denoising autoencoder"""

import pytest
import numpy as np

from raresim.similarity_methods.autoencoder.methods import (
    build_vocabulary,
    terms_to_vector,
    corrupt_vector,
    relu,
    sigmoid,
    DenoisingAutoencoder,
    euclidean_similarity,
)


# Fixtures 

def _disease_profiles() -> dict[str, dict]:
    return {
        "ORPHA:1": {"propagated_hpo_terms": ["HP:0001250", "HP:0000118"]},
        "ORPHA:2": {"propagated_hpo_terms": ["HP:0004322", "HP:0000118"]},
        "ORPHA:3": {"propagated_hpo_terms": ["HP:0001250", "HP:0004322"]},
    }


def _vocab() -> list[str]:
    return build_vocabulary(_disease_profiles())


def _term_to_idx(vocab: list[str]) -> dict[str, int]:
    return {term: i for i, term in enumerate(vocab)}


# build_vocabulary

def test_build_vocabulary_sorted() -> None:
    """Vocabulary should be sorted."""
    vocab = build_vocabulary(_disease_profiles())
    assert vocab == sorted(vocab)


def test_build_vocabulary_unique() -> None:
    """Vocabulary should contain no duplicates."""
    vocab = build_vocabulary(_disease_profiles())
    assert len(vocab) == len(set(vocab))


def test_build_vocabulary_contains_all_terms() -> None:
    """All HPO terms across profiles should appear in the vocabulary."""
    vocab = build_vocabulary(_disease_profiles())
    assert "HP:0001250" in vocab
    assert "HP:0004322" in vocab
    assert "HP:0000118" in vocab


def test_build_vocabulary_empty_profiles() -> None:
    """Empty disease profiles should produce an empty vocabulary."""
    assert build_vocabulary({}) == []


# terms_to_vector 

def test_terms_to_vector_shape() -> None:
    """Vector should have length equal to vocabulary size."""
    vocab = _vocab()
    idx = _term_to_idx(vocab)
    vec = terms_to_vector({"HP:0001250"}, vocab, idx)
    assert vec.shape == (len(vocab),)


def test_terms_to_vector_known_term_is_one() -> None:
    """A term in the vocabulary should be 1.0 in the vector."""
    vocab = _vocab()
    idx = _term_to_idx(vocab)
    vec = terms_to_vector({"HP:0001250"}, vocab, idx)
    assert vec[idx["HP:0001250"]] == pytest.approx(1.0)


def test_terms_to_vector_unknown_term_is_zero() -> None:
    """A term not in the vocabulary should leave all positions 0."""
    vocab = _vocab()
    idx = _term_to_idx(vocab)
    vec = terms_to_vector({"HP:9999999"}, vocab, idx)
    assert vec.sum() == pytest.approx(0.0)


def test_terms_to_vector_empty_terms() -> None:
    """Empty term set should produce an all-zero vector."""
    vocab = _vocab()
    idx = _term_to_idx(vocab)
    vec = terms_to_vector(set(), vocab, idx)
    assert vec.sum() == pytest.approx(0.0)


# corrupt_vector 

def test_corrupt_vector_drops_present_terms() -> None:
    """Corruption should reduce the number of present (1.0) terms."""
    vec = np.ones(100, dtype=np.float32)
    corrupted = corrupt_vector(vec, noise_rate=0.5, false_positive_rate=0.0)
    assert corrupted.sum() < vec.sum()


def test_corrupt_vector_adds_false_positives() -> None:
    """Corruption should add some 1.0 entries when false_positive_rate > 0."""
    vec = np.zeros(100, dtype=np.float32)
    corrupted = corrupt_vector(vec, noise_rate=0.0, false_positive_rate=0.3)
    assert corrupted.sum() > 0.0


def test_corrupt_vector_no_corruption() -> None:
    """With zero rates, the vector should be unchanged."""
    vec = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    corrupted = corrupt_vector(vec, noise_rate=0.0, false_positive_rate=0.0)
    np.testing.assert_array_equal(vec, corrupted)


def test_corrupt_vector_does_not_modify_original() -> None:
    """corrupt_vector should not modify the input vector."""
    vec = np.ones(10, dtype=np.float32)
    original = vec.copy()
    corrupt_vector(vec, noise_rate=0.5, false_positive_rate=0.1)
    np.testing.assert_array_equal(vec, original)


# Activations 

def test_relu_negative_to_zero() -> None:
    """ReLU should map negative values to 0."""
    x = np.array([-1.0, -0.5, 0.0, 1.0])
    result = relu(x)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0])


def test_sigmoid_bounded() -> None:
    """Sigmoid output should be in (0, 1) for reasonable inputs."""
    x = np.array([-10.0, 0.0, 10.0])
    result = sigmoid(x)
    assert (result > 0.0).all()
    assert (result < 1.0).all()


def test_sigmoid_midpoint() -> None:
    """Sigmoid(0) should be 0.5."""
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


# DenoisingAutoencoder 

def _small_ae() -> DenoisingAutoencoder:
    return DenoisingAutoencoder(vocab_size=10, hidden_dim=8, latent_dim=4)


def test_encode_output_shape() -> None:
    """Encoder output should have latent_dim dimensions."""
    ae = _small_ae()
    x = np.random.rand(10).astype(np.float32)
    latent = ae.encode(x)
    assert latent.shape == (4,)


def test_forward_output_shape() -> None:
    """Forward pass output should match vocab_size."""
    ae = _small_ae()
    x = np.random.rand(1, 10).astype(np.float32)
    out, cache = ae.forward(x)
    assert out.shape == (1, 10)


def test_forward_output_bounded() -> None:
    """Forward pass output (sigmoid decoder) should be in (0, 1)."""
    ae = _small_ae()
    x = np.random.rand(1, 10).astype(np.float32)
    out, _ = ae.forward(x)
    assert (out > 0.0).all()
    assert (out < 1.0).all()


def test_backward_returns_loss() -> None:
    """Backward pass should return a positive loss value."""
    ae = _small_ae()
    x = np.random.rand(1, 10).astype(np.float32)
    out, cache = ae.forward(x)
    loss = ae.backward(x, cache)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_train_reduces_loss() -> None:
    """Training for several epochs should reduce loss."""
    ae = DenoisingAutoencoder(vocab_size=20, hidden_dim=16, latent_dim=8)
    vectors = np.random.rand(50, 20).astype(np.float32)
    losses = ae.train(vectors, epochs=20, batch_size=10, print_every=100)
    assert losses[-1] < losses[0]


def test_save_and_load(tmp_path) -> None:
    """Saved and loaded autoencoder should produce identical encode output."""
    ae = _small_ae()
    path = tmp_path / "ae_test"
    ae.save(path)
    loaded = DenoisingAutoencoder.load(str(path) + ".npz")
    x = np.random.rand(10).astype(np.float32)
    np.testing.assert_array_almost_equal(ae.encode(x), loaded.encode(x))


# euclidean_similarity

def test_euclidean_similarity_identical_vectors() -> None:
    """Euclidean similarity of identical vectors should be 1.0."""
    vec = np.array([1.0, 2.0, 3.0])
    assert euclidean_similarity(vec, vec) == pytest.approx(1.0)


def test_euclidean_similarity_bounded() -> None:
    """Euclidean similarity should be in (0, 1]."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    score = euclidean_similarity(a, b)
    assert 0.0 < score <= 1.0


def test_euclidean_similarity_decreases_with_distance() -> None:
    """More distant vectors should have lower similarity."""
    origin = np.zeros(3)
    close = np.array([0.1, 0.0, 0.0])
    far = np.array([10.0, 0.0, 0.0])
    assert euclidean_similarity(origin, close) > euclidean_similarity(origin, far)


def test_euclidean_similarity_formula() -> None:
    """similarity = 1 / (1 + L2 distance)."""
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])  # L2 distance = 5
    assert euclidean_similarity(a, b) == pytest.approx(1.0 / 6.0)
