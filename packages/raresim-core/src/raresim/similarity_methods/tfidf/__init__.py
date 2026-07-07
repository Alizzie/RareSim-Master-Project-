"""This module provides an interface for running TF-IDF similarity methods."""
from raresim.similarity_methods.tfidf.pipeline import run
from raresim.similarity_methods.tfidf.config import ALL_METHODS as METHOD_NAMES

__all__ = ["run", "METHOD_NAMES"]
