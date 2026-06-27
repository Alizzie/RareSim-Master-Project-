"""Configuration for set-based similarity methods."""

from typing import Callable
from raresim.utils.similarity_math import TermInput
from raresim.utils.paths import SIMILARITY_DIR
from raresim.similarity_methods.set_based.methods import (
    jaccard_similarity,
    dice_similarity,
    overlap_coefficient,
    cosine_similarity,
)

SETBASED_DIR = SIMILARITY_DIR / "set_based"
PIPELINE_NAME = "set_based"

METHOD_MAP = {
    "set_cosine": cosine_similarity,
    "set_jaccard": jaccard_similarity,
    "set_overlap": overlap_coefficient,
    "set_dice": dice_similarity,
}

ALL_METHODS = list(METHOD_MAP.keys())
