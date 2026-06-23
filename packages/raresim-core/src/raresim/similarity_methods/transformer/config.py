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

from raresim.utils.paths import TRANSFORMER_DIR

CACHE_ROOT = TRANSFORMER_DIR / "cache"

TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

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
