"""Configuration file for the project, defining paths, constants, and settings."""

from shared.paths import PROJECT_ROOT

ONTOLOGY_DIR = PROJECT_ROOT / "ontologies" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPO_PATH = ONTOLOGY_DIR / "hpo.owl"
ORDO_PATH = ONTOLOGY_DIR / "ordo.owl"
MONDO_PATH = ONTOLOGY_DIR / "mondo_rare.owl"
HOOM_PATH = ONTOLOGY_DIR / "hoom.owl"
HPOA_PATH = ONTOLOGY_DIR / "phenotype.hpoa.owl"

ORPHADATA_PRODUCT4_PATH = ONTOLOGY_DIR / "en_product4_HPO.xml"

MONARCH_DISEASE_TO_HPO_PATH = (
    ONTOLOGY_DIR / "disease_to_phenotypic_feature_association.all.tsv.gz"
)

ONTOLOGY_PATHS = {
    "hpo": HPO_PATH,
    "ordo": ORDO_PATH,
    "mondo": MONDO_PATH,
    "hoom": HOOM_PATH,
    "hpoa": HPOA_PATH,
    "orphadata_product4": ORPHADATA_PRODUCT4_PATH,
    "monarch_disease_hpo": MONARCH_DISEASE_TO_HPO_PATH,
}

CANONICAL_DISEASE_NAMESPACE = "ORPHA"
APPLY_TRUE_PATH_RULE = True
MIN_DISEASE_HPO_TERMS = 1

EXAMPLE_PATIENT = {
    "patient_id": "patient_001",
    "raw_text": ("Patient with developmental delay, cerebellar ataxia, " "and anemia."),
    "hpo_terms": ["HP:0001263", "HP:0002470", "HP:0001903"],
}

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "raresim123"
NEO4J_DATABASE = "neo4j"

SHARED_DIR = OUTPUT_DIR

# ── Phenotype extraction settings ─────────────────────────────────────────────
 
# Words that indicate negation in clinical text
NEGATION_WORDS = {
    "no",
    "not",
    "without",
    "denies",
    "denied",
    "negative for",
    "absence of",
}
 
# Window size (characters) to look back for negation words
NEGATION_WINDOW_SIZE = 50
 
# HPO IDs that are structural/metadata terms, not phenotypes
# Filtered out from all extraction results
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
 
# current extraction methods
# "dictionary"     — exact HPO label matching (fast baseline)
# "synonyms"       — dictionary + HPO synonym expansion
# "biomedical_ner" — d4data transformer NER + HPO label lookup
EXTRACTION_METHODS = ["dictionary", "synonyms", "biomedical_ner"]
 
# d4data biomedical NER model
BIOMEDICAL_NER_MODEL = "d4data/biomedical-ner-all"
 
# Minimum NER confidence threshold
BIOMEDICAL_NER_MIN_CONFIDENCE = 0.6
