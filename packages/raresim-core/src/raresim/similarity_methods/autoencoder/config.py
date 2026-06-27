"""Configuration for the denoising autoencoder similarity method."""

from raresim.utils.paths import SIMILARITY_DIR

PIPELINE_NAME = "autoencoder"
METHOD_NAME = "denoising_autoencoder"
ALL_METHOD = [METHOD_NAME]

AUTOENCODER_DIR = SIMILARITY_DIR / PIPELINE_NAME
MODEL_CACHE_DIR = AUTOENCODER_DIR / "model_cache"


HIDDEN_DIM = 512  # encoder/decoder hidden layer size
LATENT_DIM = 128  # compressed latent representation size

LEARNING_RATE = 0.01  # SGD learning rate
MOMENTUM = 0.9  # SGD momentum
NOISE_RATE = 0.2  # fraction of present HPO terms to drop during training
FALSE_POSITIVE_RATE = 0.05  # fraction of absent HPO terms to add during training
EPOCHS = 50  # training epochs
BATCH_SIZE = 64  # minibatch size
