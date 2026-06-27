"""
Configuration for the transformer-based disease retrieval pipeline.

Models:
- PubMedBERT    : biomedical encoder trained on PubMed abstracts
- ClinicalBERT  : trained on clinical notes
- MiniLM        : lightweight general sentence transformer
- SapBERT       : trained for biomedical entity normalization
- BioBERT        : trained on PubMed abstracts and PMC full-text articles

All models are encoder-only (not generative) and produce fixed-size
embeddings used for cosine similarity ranking.
"""

from raresim.utils.paths import SIMILARITY_DIR

PIPELINE_NAME = "transformer"

METHOD_NAME = "transformer_cosine"
ALL_METHODS = [METHOD_NAME]

CACHE_ROOT = SIMILARITY_DIR / PIPELINE_NAME / "cache"
TRANSFORMER_DIR = SIMILARITY_DIR / PIPELINE_NAME

MODEL_LIST = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-MiniLM-L6-v2",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "dmis-lab/biobert-v1.1",
]

# Use this when the frontend should run only one transformer model.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_LIST = [DEFAULT_MODEL]

# Sentence transformer models — use SentenceTransformer library
SENTENCE_TRANSFORMER_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
}

MAX_LENGTH = 128
BATCH_SIZE = 16
CANDIDATE_POOL_SIZE = 200
TOP_K = 10

TEXT_PREVIEW_LENGTH = 300
CANDIDATE_POOL_SIZE = 200
