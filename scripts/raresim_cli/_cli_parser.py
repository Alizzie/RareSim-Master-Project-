"""Command-line argument parsing for the RareSim CLI application."""

import argparse
from pathlib import Path
from _constants import ALL_METHODS, EXTRACTION_METHODS, DEFAULTS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rare disease similarity pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input mode  ───────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text",
        type=str,
        default=None,
        metavar="CLINICAL_TEXT",
        help="Raw clinical text — extract HPO terms then run similarity",
    )
    input_group.add_argument(
        "--hpo",
        type=str,
        default=None,
        metavar="HP:XXXXXXX,...",
        help="Comma-separated HPO term IDs — skip extraction, run similarity directly",
    )
    input_group.add_argument(
        "--patient",
        type=Path,
        default=None,
        help="Path to patient JSON file with pre-extracted HPO terms",
    )
    input_group.add_argument(
        "--defaults",
        action="store_true",
        help="Skip all prompts: use example patient and all methods",
    )

    # ── Extraction settings (only used with --text) ───────────────────────────
    parser.add_argument(
        "--extraction-methods",
        nargs="+",
        default=["dictionary", "fast_hpo_cr"],
        choices=EXTRACTION_METHODS,
        help="Phenotype extraction methods (only used with --text, default: dictionary fast_hpo_cr)",
    )

    # ── Similarity settings ───────────────────────────────────────────────────
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        help="Similarity methods to run (default: all)",
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

    return parser.parse_args()
