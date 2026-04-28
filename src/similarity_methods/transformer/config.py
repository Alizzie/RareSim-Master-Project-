from pathlib import Path

"""
Configuration for the transformer-based disease retrieval pipeline.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SHARED_DIR = OUTPUTS_DIR / "shared"
TRANSFORMER_DIR = OUTPUTS_DIR / "transformer"
CACHE_ROOT = TRANSFORMER_DIR / "cache"

TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_LIST = [
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "emilyalsentzer/Bio_ClinicalBERT",
    "sentence-transformers/all-MiniLM-L6-v2",
]

TOP_K = 10
MAX_LENGTH = 128
BATCH_SIZE = 16
CANDIDATE_POOL_SIZE = 200

DISEASE_PROFILES_PATH = SHARED_DIR / "disease_profiles.json"
HPO_LABELS_PATH = SHARED_DIR / "hpo_labels.json"
PATIENT_PATH = SHARED_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = SHARED_DIR / "alias_to_canonical.json"
