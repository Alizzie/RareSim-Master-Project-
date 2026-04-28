from shared.paths import SHARED_DIR, PROJECT_ROOT
from shared.result import SimilarityResult
import shared.io as io
from pathlib import Path


def check_artifacts_exist() -> None:
    """Fail fast with a clear message if build step hasn't been run."""
    required = [
        SHARED_DIR / "canonical_disease_profiles.json",
        SHARED_DIR / "hpo_labels.json",
        SHARED_DIR / "information_content.json",
        SHARED_DIR / "hpo_ancestors.json",
        SHARED_DIR / "example_patient.json",
    ]
    missing = [f.name for f in required if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing shared artifacts: {', '.join(missing)}\n"
            "Run 'python -m build_shared_artifacts' first."
        )


# -- Display helpers ----------------------------------


def print_results_table(method_name: str, results: list[SimilarityResult]) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {method_name}")
    print(f"{'─' * 64}")
    if not results:
        print("  No results.")
        return
    for row in results:
        print(
            f"  rank={row.rank:>2} | "
            f"{row.disease_id:<15} | "
            f"score={row.score:.4f} | "
            f"{row.label}"
        )


def save_results(all_results: dict) -> None:
    out_dir = PROJECT_ROOT / "outputs" / "gui"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_results.json"
    io.save_results(all_results, path)
    print(f"\nResults saved to: {path}")


# -- Prompts ----------------------------------
def prompt_patient(DEFAULTS: dict) -> dict:
    print("\nNo patient file provided.")
    print("  [1] Load from JSON file path")
    print("  [2] Use default example patient")
    choice = input("\nChoice (1/2): ").strip()

    if choice == "1":
        path = input("Path to patient JSON file: ").strip()
        return io.load_patient(Path(path))

    print(f"Using default: {DEFAULTS['patient_path'].name}")
    return io.load_patient(DEFAULTS["patient_path"])


def prompt_methods(DEFAULTS: dict, ALL_METHODS: list[str]) -> list[str]:
    print("\nAvailable methods:")
    for i, name in enumerate(ALL_METHODS, 1):
        print(f"  [{i:>2}] {name}")
    print(f"\n  Press Enter to select all ({len(ALL_METHODS)} methods)")

    raw = input("Select methods (comma-separated numbers, or Enter for all): ").strip()

    if not raw:
        return ALL_METHODS

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(ALL_METHODS):
                selected.append(ALL_METHODS[idx])

    return selected if selected else ALL_METHODS
