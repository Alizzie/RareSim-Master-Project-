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
