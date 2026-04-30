"""
Utility functions for the GUI application, including artifact checks, result display, and user prompts.
"""

from shared.paths import SHARED_DIR, PROJECT_ROOT
from shared.result import SimilarityResult
import shared.io as io
from pathlib import Path

GUI_DIR = PROJECT_ROOT / "outputs" / "gui"


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
    """Print the results for a given method."""
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


def save_results(all_results: dict, app_metadata: dict) -> None:
    """Save all results to a JSON file in the outputs directory inside gui."""
    GUI_DIR.mkdir(parents=True, exist_ok=True)
    path = GUI_DIR / "app_metadata.json"
    io.save_json(app_metadata, path)
    print(f"App metadata saved to: {path}")

    path = GUI_DIR / "all_results.json"
    io.save_results(all_results, path)
    print(f"\nResults saved to: {path}")

    for method_name, rows in all_results.items():
        path = GUI_DIR / f"{method_name}_top{len(rows)}.json"
        io.save_individual_results(rows, path)
    print(f"Individual method results saved to: {GUI_DIR}")


# -- Prompts ----------------------------------
def prompt_patient(defaults: dict, hpo_labels: dict) -> dict:
    """Prompt the user to select a patient profile, either from a JSON file or using the default example."""
    print("\nNo patient file provided.")
    print("  [1] Load from JSON file path")
    print("  [2] Use default example patient")
    choice = input("\nChoice (1/2): ").strip()

    if choice == "1":
        path = input("Path to patient JSON file: ").strip()
        return io.load_patient_with_extraction(Path(path), hpo_labels)

    print(f"Using default: {defaults['patient_path'].name}")
    return io.load_patient_with_extraction(defaults["patient_path"], hpo_labels)


def prompt_methods(all_methods: list[str]) -> list[str]:
    """Prompt the user to select which similarity methods to run, either by choosing from a list or using all methods."""
    print("\nAvailable methods:")
    for i, name in enumerate(all_methods, 1):
        print(f"  [{i:>2}] {name}")
    print(f"\n  Press Enter to select all ({len(all_methods)} methods)")

    raw = input("Select methods (comma-separated numbers, or Enter for all): ").strip()

    if not raw:
        return all_methods

    selected = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(all_methods):
                selected.append(all_methods[idx])

    return selected if selected else all_methods
