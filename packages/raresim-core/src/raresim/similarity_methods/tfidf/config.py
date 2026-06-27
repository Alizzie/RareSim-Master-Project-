"""Configuration for the TF-IDF similarity pipeline."""

from raresim.utils.paths import SIMILARITY_DIR

PIPELINE_NAME = "tfidf"
TFIDF_DIR = SIMILARITY_DIR / "tfidf"


# HPO-based TF-IDF: compares HPO term sets weighted by IDF over the corpus.
# Binary TF — term either present or not. IDF alone carries the signal.
METHOD_HPO = "tfidf_hpo"

# Text-based TF-IDF: compares tokenized clinical words.
# Patient raw_text vs disease merged_description.
# True TF (word count) × IDF over the description corpus.
METHOD_TEXT = "tfidf_text"

# Hybrid: patient HPO labels (strings) vs disease description text.
METHOD_HYBRID = "tfidf_hybrid"

# HPO-Labels-based TFIDF: patient HPO labels (strings) vs disease HPO labels (strings).
METHOD_HPO_LABELS = "tfidf_hpo_labels"

ALL_METHODS = [METHOD_HPO, METHOD_TEXT, METHOD_HYBRID, METHOD_HPO_LABELS]

# IDF terms whose weight falls below this are flagged as common noise.
LOW_IDF_THRESHOLD = 0.5

# Top N IDF-weighted matched terms to surface in the explanation.
TOP_N_IDF_MATCHES = 10

# Minimum token length — single characters and two-letter words are noise.
MIN_TOKEN_LENGTH = 3

# Top N TF-IDF weighted tokens to surface in the explanation.
TOP_N_TEXT_MATCHES = 10

# Whether to use disease merged_description or label as the disease document.
# "description" gives richer text; "label" is a reliable fallback.
DISEASE_TEXT_FIELD = "merged_description"

# disease descriptions with fewer than this many tokens are considered sparse
SPARSE_DISEASE_THRESHOLD = 5
