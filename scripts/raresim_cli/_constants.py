"""Constants for the RareSim CLI application."""

from raresim.utils.paths import EXAMPLE_PATIENT_PATH

SEMANTIC_METHODS = [
    "semantic_resnik_bma",
    "semantic_lin_bma",
    "semantic_jiang_conrath_bma",
]

SET_BASED_METHODS = [
    "set_cosine",
    "set_jaccard",
    "set_dice",
    "set_overlap",
]

TFIDF_METHODS = ["tfidf"]

TRANSFORMER_METHODS = ["transformer"]

LLM_METHODS = ["llm"]

ALL_METHODS = (
    SEMANTIC_METHODS
    + SET_BASED_METHODS
    + TFIDF_METHODS
    + TRANSFORMER_METHODS
    + LLM_METHODS
)

EXTRACTION_METHODS = [
    "dictionary",
    "biomedical_ner",
    "fast_hpo_cr",
    "chatgpt",
    "phenobrain_api",
]

DEFAULTS = {
    "patient_path": EXAMPLE_PATIENT_PATH,
    "methods": ALL_METHODS,
    "top_k": 10,
    "use_propagated_terms": True,
    "ic_threshold": 1.5,
    "use_canonical_profiles": True,
}
