"""Configuration for semantic similarity methods."""

from raresim.utils.paths import SIMILARITY_DIR
from raresim.similarity_methods.semantic.methods import (
    resnik_similarity,
    lin_similarity,
    jiang_conrath_similarity,
)

SEMANTIC_DIR = SIMILARITY_DIR / "semantic"
PIPELINE_NAME = "semantic"

# BMA methods: pairwise term-to-term comparison averaged bidirectionally
BMA_METHODS = {
    "semantic_resnik_bma": resnik_similarity,
    "semantic_lin_bma": lin_similarity,
    "semantic_jiang_conrath_bma": jiang_conrath_similarity,
}

ALL_METHODS = list(BMA_METHODS)

# Threshold below which a patient term's best BMA score is considered "weak"
WEAK_MATCH_THRESHOLD = 0.3
