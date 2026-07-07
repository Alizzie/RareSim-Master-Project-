"""This module provides an interface for running HPO2Vec-based similarity methods."""
from raresim.similarity_methods.hpo2vec.pipeline import run
from raresim.similarity_methods.hpo2vec.config import (
    ALL_METHODS as METHOD_NAMES,
    MODEL_CACHE_DIR,
    PIPELINE_NAME,
)

__all__ = ["run", "METHOD_NAMES", "MODEL_CACHE_DIR", "PIPELINE_NAME"]
