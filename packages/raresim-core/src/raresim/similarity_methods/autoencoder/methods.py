"""
Denoising Autoencoder methods for HPO-based disease similarity
"""

from pathlib import Path
import numpy as np

from raresim.similarity_methods.autoencoder.config import (
    FALSE_POSITIVE_RATE,
    NOISE_RATE,
)


def build_vocabulary(
    disease_profiles: dict[str, dict],
    terms_key: str = "propagated_hpo_terms",
) -> list[str]:
    """Build a sorted list of unique HPO terms across all disease profiles."""
    vocab = set()
    for profile in disease_profiles.values():
        vocab.update(profile.get(terms_key, []))
    return sorted(vocab)


def terms_to_vector(
    terms: set[str],
    vocab: list[str],
    term_to_idx: dict[str, int],
) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for term in terms:
        if term in term_to_idx:
            vec[term_to_idx[term]] = 1.0
    return vec


def corrupt_vector(
    vec: np.ndarray,
    noise_rate: float = NOISE_RATE,
    false_positive_rate: float = FALSE_POSITIVE_RATE,
) -> np.ndarray:
    """
    Corrupt a binary HPO vector:
      - Drop some present terms (masking noise)
      - Add some absent terms as false positives
    Both corruptions force the encoder to learn structure, not memorize.
    """
    corrupted = vec.copy()
    present = np.where(corrupted > 0)[0]
    absent = np.where(corrupted == 0)[0]

    n_drop = int(len(present) * noise_rate)
    if n_drop > 0:
        corrupted[np.random.choice(present, n_drop, replace=False)] = 0.0

    n_add = int(len(absent) * false_positive_rate)
    if n_add > 0:
        corrupted[np.random.choice(absent, n_add, replace=False)] = 1.0

    return corrupted


# Activations


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_grad(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)


class DenoisingAutoencoder:
    """
    Encoder uses ReLU this time
    Decoder output uses sigmoid (probabilities for BCE loss)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        noise_rate: float = NOISE_RATE,
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.momentum = momentum
        self.noise_rate = noise_rate

        # init for ReLU layers (encoder), Xavier for sigmoid (decoder)
        self.W1 = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * np.sqrt(
            2.0 / vocab_size
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * np.sqrt(
            2.0 / hidden_dim
        )
        self.b2 = np.zeros(latent_dim, dtype=np.float32)

        self.W3 = np.random.randn(latent_dim, hidden_dim).astype(np.float32) * np.sqrt(
            2.0 / (latent_dim + hidden_dim)
        )
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)

        self.W4 = np.random.randn(hidden_dim, vocab_size).astype(np.float32) * np.sqrt(
            2.0 / (hidden_dim + vocab_size)
        )
        self.b4 = np.zeros(vocab_size, dtype=np.float32)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)
        self.vW4 = np.zeros_like(self.W4)
        self.vb4 = np.zeros_like(self.b4)

    def encode(self, x: np.ndarray) -> np.ndarray:
        h1 = relu(x @ self.W1 + self.b1)
        latent = relu(h1 @ self.W2 + self.b2)
        return latent

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid decoder, outputs stay in [0,1] for BCE."""
        h3 = sigmoid(z @ self.W3 + self.b3)
        out = sigmoid(h3 @ self.W4 + self.b4)
        return out

    def forward(self, x_corrupted: np.ndarray) -> tuple[np.ndarray, dict]:
        h1 = relu(x_corrupted @ self.W1 + self.b1)
        z = relu(h1 @ self.W2 + self.b2)
        h3 = sigmoid(z @ self.W3 + self.b3)
        out = sigmoid(h3 @ self.W4 + self.b4)
        cache = {"x": x_corrupted, "h1": h1, "z": z, "h3": h3, "out": out}
        return out, cache

    def backward(self, x_clean: np.ndarray, cache: dict) -> float:
        x, h1, z, h3, out = (
            cache["x"],
            cache["h1"],
            cache["z"],
            cache["h3"],
            cache["out"],
        )
        batch_size = x.shape[0]

        eps = 1e-8
        loss = -np.mean(
            x_clean * np.log(out + eps) + (1 - x_clean) * np.log(1 - out + eps)
        )

        # Decoder backward (sigmoid layers)
        d_out = (out - x_clean) / batch_size
        dW4 = h3.T @ (d_out * sigmoid_grad(out))
        db4 = np.sum(d_out * sigmoid_grad(out), axis=0)

        d_h3 = (d_out * sigmoid_grad(out)) @ self.W4.T
        dW3 = z.T @ (d_h3 * sigmoid_grad(h3))
        db3 = np.sum(d_h3 * sigmoid_grad(h3), axis=0)

        # Encoder backward (ReLU layers this)
        d_z = (d_h3 * sigmoid_grad(h3)) @ self.W3.T
        dW2 = h1.T @ (d_z * relu_grad(z))
        db2 = np.sum(d_z * relu_grad(z), axis=0)

        d_h1 = (d_z * relu_grad(z)) @ self.W2.T
        dW1 = x.T @ (d_h1 * relu_grad(h1))
        db1 = np.sum(d_h1 * relu_grad(h1), axis=0)

        for W, b, dW, db, vW, vb in [
            (self.W1, self.b1, dW1, db1, self.vW1, self.vb1),
            (self.W2, self.b2, dW2, db2, self.vW2, self.vb2),
            (self.W3, self.b3, dW3, db3, self.vW3, self.vb3),
            (self.W4, self.b4, dW4, db4, self.vW4, self.vb4),
        ]:
            vW[:] = self.momentum * vW - self.lr * dW
            vb[:] = self.momentum * vb - self.lr * db
            W += vW
            b += vb

        return float(loss)

    def train(
        self,
        vectors: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        print_every: int = 10,
    ) -> list[float]:
        n = len(vectors)
        epoch_losses = []

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            batch_losses = []

            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_clean = vectors[batch_idx]
                x_corrupted = np.array(
                    [corrupt_vector(v, self.noise_rate) for v in x_clean]
                )
                out, cache = self.forward(x_corrupted)
                loss = self.backward(x_clean, cache)
                batch_losses.append(loss)

            avg_loss = float(np.mean(batch_losses))
            epoch_losses.append(avg_loss)

            if epoch % print_every == 0 or epoch == 1:
                print(f"  epoch {epoch:>3}/{epochs} | loss={avg_loss:.4f}")

        return epoch_losses

    def save(self, path: Path) -> None:
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            W4=self.W4,
            b4=self.b4,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )

    @classmethod
    def load(cls, path: Path) -> "DenoisingAutoencoder":
        data = np.load(path)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            hidden_dim=int(data["hidden_dim"]),
            latent_dim=int(data["latent_dim"]),
        )
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        model.W3, model.b3 = data["W3"], data["b3"]
        model.W4, model.b4 = data["W4"], data["b4"]
        return model


def euclidean_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Convert L2 distance to a similarity score in [0, 1]."""
    dist = np.linalg.norm(vec_a - vec_b)
    return float(1.0 / (1.0 + dist))
