"""This module provides an interface for running autoencoder-based similarity methods."""
from raresim.similarity_methods.autoencoder.pipeline import run
from raresim.similarity_methods.autoencoder.config import ALL_METHODS as METHOD_NAMES

__all__ = ["run", "METHOD_NAMES"]
