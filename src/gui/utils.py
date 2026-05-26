"""
Utility functions for the GUI application, including artifact checks, result display, and user prompts.
"""

from shared.paths import SHARED_DIR, PROJECT_ROOT
from shared.result import AppMetadata, MethodResults
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


def print_app_metadata(app_metadata: AppMetadata) -> None:
    """Print the app-level data and patient summary."""
    print(f"\n{'─' * 64}")
    print("  Run summary")
    print(f"{'─' * 64}")
    for key, val in app_metadata.to_dict().items():
        print(f"  {key}: {val}")


def print_results_table(method_results: MethodResults) -> None:
    """Print ranked results for a single method."""
    meta = method_results.metadata
    print(f"\n{'─' * 64}")
    print(f"  {meta.method_name}  ({meta.computation_time:.3f}s)")
    print(f"{'─' * 64}")
    if not method_results.rankings:
        print("  No results.")
        return
    for r in method_results.rankings:
        print(
            f"  rank={r.rank:>2} | "
            f"{r.disease_id:<15} | "
            f"score={r.score:.4f} | "
            f"{r.label}"
        )


def print_raw_results(method_name: str, results: list[dict]) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {method_name}  (raw output)")
    print(f"{'─' * 64}")
    if not results:
        print("  No results.")
        return
    for r in results:
        if not isinstance(r, dict):
            print(f"  {r}")
            continue
        disease_id = r.get("canonical_disease_id") or r.get("ordo_id", "")
        label = r.get("label") or r.get("disease_name", "")
        score = r.get("score") or r.get("confidence", "")
        rank = r.get("rank", "?")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  rank={rank:>2} | {disease_id:<15} | score={score_str} | {label}")


def save_results(
    all_results: dict[str, MethodResults], app_metadata: AppMetadata
) -> None:
    """Save all results to a JSON file in the outputs directory inside gui."""
    GUI_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = GUI_DIR / "app_metadata.json"
    io.save_json(app_metadata.to_dict(), meta_path)
    print(f"App metadata saved to: {meta_path}")

    combined_path = GUI_DIR / "all_results.json"
    io.save_results(all_results, combined_path)
    print(f"\nResults saved to: {combined_path}")

    # one file per method
    io.save_individual_results(all_results, GUI_DIR)
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
