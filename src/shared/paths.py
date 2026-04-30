from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Find the project root directory by looking for a marker file or directory."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root containing '{marker}'")


PROJECT_ROOT = find_project_root()
SHARED_DIR = PROJECT_ROOT / "outputs" / "shared"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"

TRANSFORMER_DIR = OUTPUTS_DIR / "transformer"
LLM_DIR = OUTPUTS_DIR / "llm"

DISEASE_PROFILES_PATH = SHARED_DIR / "canonical_disease_profiles.json"
HPO_LABELS_PATH = SHARED_DIR / "hpo_labels.json"
PATIENT_PATH = SHARED_DIR / "example_patient.json"
ALIAS_TO_CANONICAL_PATH = SHARED_DIR / "alias_to_canonical.json"
INFORMATION_CONTENT_PATH = SHARED_DIR / "information_content.json"
HPO_ANCESTORS_PATH = SHARED_DIR / "hpo_ancestors.json"

PHENOTYPE_DIR = OUTPUTS_DIR / "phenotype"
OUTPUT_PATIENT_PATH = PHENOTYPE_DIR / "example_patient_extracted.json"
OUTPUT_EXTRACTION_PATH = PHENOTYPE_DIR / "example_patient_hpo_extraction.json"
