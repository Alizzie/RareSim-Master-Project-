"""
Terminal interface for the rare disease similarity pipeline.

Usage:
    # Skip all prompts — use example patient, all methods:
    python src/gui/app.py --defaults

    # Pass a patient JSON file:
    python src/gui/app.py --patient outputs/shared/example_patient.json

    # Select specific methods:
    python src/gui/app.py --patient outputs/shared/example_patient.json --methods semantic_resnik tfidf

    # Change top-k:
    python src/gui/app.py --defaults --top-k 5
"""

import argparse
from pathlib import Path
from shared.paths import SHARED_DIR
from shared.pipeline import PipelineConfig
from shared.context import AppContext
from shared.io import load_patient
from gui.utils import (
    check_artifacts_exist,
    print_results_table,
    save_results,
    prompt_patient,
    prompt_methods,
)

from similarity_methods.set_based.pipeline import run as run_set_based

ALL_METHODS = [
    "semantic_resnik",
    "semantic_lin",
    "semantic_jiang_conrath",
    "semantic_simgic",
    "semantic_icto",
    "semantic_jaccard",
    "set_cosine",
    "set_jaccard",
    "set_dice",
    "set_overlap",
    "tfidf",
    "transformer",
]

DEFAULTS = {
    "patient_path": SHARED_DIR / "example_patient.json",
    "methods": ALL_METHODS,
    "top_k": 10,
    "use_propagated_terms": True,
    "ic_threshold": 1.5,
    "use_canonical_profiles": True,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rare disease similarity pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--patient",
        type=Path,
        help="Path to patient JSON file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        help="Methods to run (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULTS["top_k"],
        help=f"Number of top results per method (default: {DEFAULTS['top_k']})",
    )
    parser.add_argument(
        "--no-propagation",
        action="store_true",
        help="Use raw HPO terms instead of propagated",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=DEFAULTS["ic_threshold"],
        help=f"Minimum IC value to include a term (default: {DEFAULTS['ic_threshold']})",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Skip all prompts: use example patient and all methods",
    )
    return parser.parse_args()


# --- Entry point ----------------------------------
def main() -> None:
    """Main function to run the terminal interface."""
    check_artifacts_exist()
    args = parse_args()

    print("=" * 64)
    print("  Rare Disease Similarity Pipeline")
    print("=" * 64)

    # ── Patient ───────────────────────────────────────────────────────────────
    if args.patient:
        patient = load_patient(args.patient)
        print(f"\nPatient: {args.patient.name}")
    elif args.defaults:
        patient = load_patient(DEFAULTS["patient_path"])
        print(f"\nPatient: {DEFAULTS['patient_path'].name} (default)")
    else:
        patient = prompt_patient(DEFAULTS)

    print(f"  HPO terms: {len(patient.hpo_terms)}")
    print(f"  Propagated terms: {len(patient.propagated_hpo_terms)}")

    # ── Methods ───────────────────────────────────────────────────────────────
    if args.methods:
        selected = args.methods
    elif args.defaults:
        selected = ALL_METHODS
    else:
        selected = prompt_methods(DEFAULTS, ALL_METHODS)

    print(f"\nSelected methods ({len(selected)}): {', '.join(selected)}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = PipelineConfig(
        top_k=args.top_k,
        use_propagated_terms=not args.no_propagation,
        ic_threshold=args.ic_threshold,
        use_canonical_profiles=DEFAULTS["use_canonical_profiles"],
    )
    print(
        f"Config: top_k={config.top_k}, "
        f"propagated={config.use_propagated_terms}, "
        f"ic_threshold={config.ic_threshold}"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\nRunning pipeline...")
    all_results = {}
    ctx = AppContext.load(patient, config.use_canonical_profiles)

    print(f"\n Data summary:")
    summary = ctx.app_metadata.to_dict()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if any(m.startswith("set_") for m in selected):
        print("  Running set-based methods...")
        all_results.update(
            run_set_based(
                patient,
                selected,
                config,
                ctx,
            )
        )

    # ── Display ───────────────────────────────────────────────────────────────
    for method_name, results in all_results.items():
        print_results_table(method_name, results)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(all_results)


if __name__ == "__main__":
    main()
