from pathlib import Path

"""Configuration file for the project, defining paths, constants, and settings."""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_DIR = PROJECT_ROOT / "ontologies" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPO_PATH = ONTOLOGY_DIR / "hpo.owl"
ORDO_PATH = ONTOLOGY_DIR / "ordo.owl"
MONDO_PATH = ONTOLOGY_DIR / "mondo_rare.owl"
HOOM_PATH = ONTOLOGY_DIR / "hoom.owl"
HPOA_PATH = ONTOLOGY_DIR / "phenotype.hpoa"

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