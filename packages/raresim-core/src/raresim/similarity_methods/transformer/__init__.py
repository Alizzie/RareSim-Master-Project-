"""This module provides an interface for running transformer-based similarity methods."""
from raresim.similarity_methods.transformer.pipeline import run
from raresim.similarity_methods.transformer.config import (
    ALL_METHODS as METHOD_NAMES,
    MODEL_LIST,
)

__all__ = ["run", "METHOD_NAMES", "MODEL_LIST"]
