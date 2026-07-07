"""Configuration for set-based similarity methods."""

from raresim.utils.paths import SIMILARITY_DIR
from raresim.similarity_methods.set_based.methods import (
    jaccard_similarity,
    dice_similarity,
    overlap_coefficient,
    cosine_similarity,
    jaccard_with_negative_penalty,
)

SETBASED_DIR = SIMILARITY_DIR / "set_based"
PIPELINE_NAME = "set_based"

NEGATIVE_PENALTY_WEIGHT = 0.5

METHOD_MAP = {
    "set_cosine": cosine_similarity,
    "set_jaccard": jaccard_similarity,
    "set_overlap": overlap_coefficient,
    "set_dice": dice_similarity,
    "set_jaccard_penalized": jaccard_with_negative_penalty,
}

METHODS_REQUIRING_EXCLUSIONS = {"set_jaccard_penalized"}

ALL_METHODS = list(METHOD_MAP.keys())
