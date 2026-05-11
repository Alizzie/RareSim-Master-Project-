"""
Configuration for the transformer-based disease retrieval pipeline.
"""

from shared.paths import PROJECT_ROOT

TRANSFORMER_DIR = PROJECT_ROOT / "outputs" / "transformer"
CACHE_ROOT = TRANSFORMER_DIR / "cache"

TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_LIST = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-MiniLM-L6-v2",
]

MAX_LENGTH = 128
BATCH_SIZE = 16
CANDIDATE_POOL_SIZE = 200

# LLM explanation settings
RUN_LLM_EXPLAINER = True # Set to False to skip LLM explanation step
LLM_EXPLAINER_MODEL = "mistral"
TOP_K_LLM_EXPLAIN = 10
TOP_K=10
