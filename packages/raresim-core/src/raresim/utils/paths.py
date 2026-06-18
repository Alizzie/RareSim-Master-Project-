"""Path and file constants for RareSim project."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.environ["RARESIM_ROOT"])

# dirs data
DATA_DIR = PROJECT_ROOT / "data"
ONTOLOGY_DIR = DATA_DIR / "ontologies"
DATASET_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
PATIENT_DIR = DATA_DIR / "patient_profiles"
PHENOPACKETS_DIR = DATASET_DIR / "phenopackets/raw"
STANDARDIZED_PHENOPACKETS_DIR = DATASET_DIR / "phenopackets/standardized_to_json"


# Dirs outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"
SIMILARITY_DIR = OUTPUTS_DIR / "similarity_methods"
TRANSFORMER_DIR = OUTPUTS_DIR / "transformer"
WEBAPP_DIR = OUTPUTS_DIR / "webapp"
GUI_DIR = OUTPUTS_DIR / "gui"


# Third Party
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
FAST_HPO_CR_DIR = THIRD_PARTY_DIR / "fast_hpo_cr"

# files
DISEASE_PROFILES_PATH = ARTIFACTS_DIR / "canonical_disease_profiles.json"
HPO_LABELS_PATH = ARTIFACTS_DIR / "hpo_labels.json"
PATIENT_PATH = ARTIFACTS_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = ARTIFACTS_DIR / "alias_to_canonical.json"
INFORMATION_CONTENT_PATH = ARTIFACTS_DIR / "information_content.json"
HPO_ANCESTORS_PATH = ARTIFACTS_DIR / "hpo_ancestors.json"
HPO_PARENTS_PATH = ARTIFACTS_DIR / "hpo_parents.json"
EXAMPLE_PATIENT_PATH = PATIENT_DIR / "example_patient.json"
ORPHA_MAPPING_INDEX = ARTIFACTS_DIR / "orpha_mapping_index.json"
