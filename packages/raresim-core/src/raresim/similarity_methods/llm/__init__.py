"""This module provides an interface for running LLM-based similarity methods."""
from raresim.similarity_methods.llm.pipeline import run
from raresim.similarity_methods.llm.config import LLM_MODEL_LIST

METHOD_NAMES = list(LLM_MODEL_LIST)

__all__ = ["run", "LLM_MODEL_LIST", "METHOD_NAMES"]
