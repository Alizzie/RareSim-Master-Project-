from pathlib import Path
'''Configuration file for the project, defining paths, constants, and settings.'''

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_DIR = PROJECT_ROOT / "ontologies" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPO_PATH = ONTOLOGY_DIR / "hpo.owl"
ORDO_PATH = ONTOLOGY_DIR / "ordo.owl"
MONDO_PATH = ONTOLOGY_DIR / "mondo_rare.owl"
HOOM_PATH = ONTOLOGY_DIR / "hoom.owl"
HPOA_PATH = ONTOLOGY_DIR / "phenotype.hpoa.owl"

ONTOLOGY_PATHS = {
    "hpo": HPO_PATH,
    "ordo": ORDO_PATH,
    "mondo": MONDO_PATH,
    "hoom": HOOM_PATH,
    "hpoa": HPOA_PATH,
}

CANONICAL_DISEASE_NAMESPACE = "ORPHA"
APPLY_TRUE_PATH_RULE = True
MIN_DISEASE_HPO_TERMS = 1

EXAMPLE_PATIENT = {
    "patient_id": "patient_001",
    "raw_text": (
        "Patient with developmental delay, cerebellar ataxia, "
        "and anemia."
    ),
    "hpo_terms": ["HP:0001263", "HP:0002470", "HP:0001903"],
}
