from pathlib import Path

"""
Configuration for the phenotype extraction pipeline.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SHARED_DIR = OUTPUTS_DIR / "shared"
PHENOTYPE_DIR = OUTPUTS_DIR / "phenotype"

PHENOTYPE_DIR.mkdir(parents=True, exist_ok=True)

# Input paths
HPO_LABELS_PATH = SHARED_DIR / "hpo_labels.json"
HPO_SYNONYMS_PATH = SHARED_DIR / "hpo_synonyms.json"

# Output paths
OUTPUT_PATIENT_PATH = PHENOTYPE_DIR / "example_patient_extracted.json"
OUTPUT_EXTRACTION_PATH = PHENOTYPE_DIR / "example_patient_hpo_extraction.json"

# Negation detection
NEGATION_WORDS = {
    "no",
    "not",
    "without",
    "denies",
    "denied",
    "negative for",
    "absence of",
}

NEGATION_WINDOW_SIZE = 50

# HPO IDs that are structural/metadata terms, not phenotypes
# These are filtered out from all extraction results
HPO_BLOCKLIST = {
    "HP:0000005",  # Mode of inheritance
    "HP:0000001",  # All (root)
    "HP:0000118",  # Phenotypic abnormality (too broad)
    "HP:0012823",  # Clinical modifier
    "HP:0031797",  # Clinical course
    "HP:0040279",  # Frequency
    "HP:0032316",  # Family history — not a phenotype
    "HP:0003674",  # Onset — metadata
    "HP:0012777",  # Biomarker — metadata
}

# Active extraction methods
# "dictionary"     — exact HPO label matching (fast baseline)
# "synonyms"       — dictionary + HPO synonym expansion
# "biomedical_ner" — d4data transformer NER + HPO label lookup
EXTRACTION_METHODS = ["dictionary", "synonyms", "biomedical_ner"]

# d4data biomedical NER model
BIOMEDICAL_NER_MODEL = "d4data/biomedical-ner-all"

# Minimum NER confidence threshold — detections below this are discarded
BIOMEDICAL_NER_MIN_CONFIDENCE = 0.6
