"""
All filesystem path constants for the RareSim project.

Single source of truth for where files live. Nothing else belongs here —
build behaviour constants (flags, thresholds, example data) live in
core/config.py.

Import pattern:
    from raresim.utils.paths import ARTIFACTS_DIR, HPO_PATH
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.environ["RARESIM_ROOT"])

# ── Data directories ──────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
ONTOLOGY_DIR = DATA_DIR / "ontologies"
DATASET_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
PATIENT_DIR = DATA_DIR / "patient_profiles"
PHENOPACKETS_DIR = DATASET_DIR / "phenopackets/raw"
STANDARDIZED_PHENOPACKETS_DIR = DATASET_DIR / "phenopackets/standardized_to_json"

# ── Output directories ────────────────────────────────────────────────────────

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "artifacts"
SIMILARITY_DIR = OUTPUTS_DIR / "similarity_methods"
TRANSFORMER_DIR = SIMILARITY_DIR / "transformer"
WEBAPP_DIR = OUTPUTS_DIR / "webapp"
GUI_DIR = OUTPUTS_DIR / "gui"

# ── Third party ───────────────────────────────────────────────────────────────

THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
FAST_HPO_CR_DIR = THIRD_PARTY_DIR / "fast_hpo_cr"

# ── Ontology source files (inputs to build_shared_artifacts) ──────────────────

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

# ── Built artifact files (outputs of build_shared_artifacts) ──────────────────

DISEASE_PROFILES_PATH = ARTIFACTS_DIR / "canonical_disease_profiles.json"
HPO_LABELS_PATH = ARTIFACTS_DIR / "hpo_labels.json"
PATIENT_PATH = ARTIFACTS_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = ARTIFACTS_DIR / "alias_to_canonical.json"
INFORMATION_CONTENT_PATH = ARTIFACTS_DIR / "information_content.json"
HPO_ANCESTORS_PATH = ARTIFACTS_DIR / "hpo_ancestors.json"
HPO_PARENTS_PATH = ARTIFACTS_DIR / "hpo_parents.json"
EXAMPLE_PATIENT_PATH = PATIENT_DIR / "example_patient.json"
ORPHA_MAPPING_INDEX = ARTIFACTS_DIR / "orpha_mapping_index.json"
DISEASE_PARENTS_PATH = ARTIFACTS_DIR / "disease_parents.json"
DISEASE_ANCESTORS_PATH = ARTIFACTS_DIR / "disease_ancestors.json"
DISEASE_METADATA_INDEX_PATH = ARTIFACTS_DIR / "disease_metadata_index.json"
