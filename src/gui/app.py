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

from shared.cache import save_run_cache
from shared.context import AppContext
from shared.io import load_json, load_patient, load_patient_with_extraction
from shared.paths import (
    ALIAS_TO_CANONICAL_PATH,
    HPO_LABELS_PATH,
    SHARED_DIR,
)
from shared.pipeline import PipelineConfig
from gui.utils import (
    check_artifacts_exist,
    print_results_table,
    save_results,
    prompt_patient,
    prompt_methods,
)

from similarity_methods.semantic.pipeline import run as run_semantic
from similarity_methods.set_based.pipeline import run as run_set_based
from similarity_methods.tfidf.pipeline import run as run_tfidf
from similarity_methods.transformer.pipeline import run as run_transformer
from similarity_methods.llm.pipeline import run as run_llm

# ── Method registry ───────────────────────────────────────────────────────────
SEMANTIC_METHODS = [
    "semantic_resnik_bma",
    "semantic_lin_bma",
    "semantic_jiang_conrath_bma",
]

SET_BASED_METHODS = [
    "set_cosine",
    "set_jaccard",
    "set_dice",
    "set_overlap",
]

TFIDF_METHODS = ["tfidf"]

TRANSFORMER_METHODS = ["transformer"]

LLM_METHODS = ["llm"]

ALL_METHODS = (
    SEMANTIC_METHODS
    + SET_BASED_METHODS
    + TFIDF_METHODS
    + TRANSFORMER_METHODS
    + LLM_METHODS
)

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


def print_raw_results(method_name: str, results: list[dict]) -> None:
    """Print raw dict results for transformer and LLM pipelines."""
    print(f"\n{'─' * 64}")
    print(f"  {method_name}")
    print(f"{'─' * 64}")
    if not results:
        print("  No results.")
        return
    for r in results:
        disease_id = r.get("canonical_disease_id") or r.get("ordo_id", "")
        label = r.get("label") or r.get("disease_name", "")
        score = r.get("score") or r.get("confidence", "")
        rank = r.get("rank", "?")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  rank={rank:>2} | {disease_id:<15} | score={score_str} | {label}")


def main() -> None:
    """Main function to run the terminal interface."""
    check_artifacts_exist()
    args = parse_args()

    hpo_labels = load_json(HPO_LABELS_PATH)

    print("=" * 64)
    print("  Rare Disease Similarity Pipeline")
    print("=" * 64)

    # ── Patient ───────────────────────────────────────────────────────────────
    if args.patient:
        patient = load_patient_with_extraction(args.patient, hpo_labels)
        print(f"\nPatient: {args.patient.name}")
    elif args.defaults:
        patient = load_patient_with_extraction(DEFAULTS["patient_path"], hpo_labels)
        print(f"\nPatient: {DEFAULTS['patient_path'].name} (default)")
    else:
        patient = prompt_patient(DEFAULTS, hpo_labels)

    print(f"  HPO terms: {len(patient.hpo_terms)}")
    print(f"  Propagated terms: {len(patient.propagated_hpo_terms)}")

    # ── Methods ───────────────────────────────────────────────────────────────
    if args.methods:
        selected = args.methods
    elif args.defaults:
        selected = ALL_METHODS
    else:
        selected = prompt_methods(ALL_METHODS)

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

    # ── Shared context (for SimilarityResult pipelines) ───────────────────────
    print("\nRunning pipeline...")
    all_results = {}
    all_raw_results = {}  # for transformer and LLM raw dict results

    ctx = AppContext.load(patient, config.use_canonical_profiles)

    print(f"\nData summary:")
    for key, value in ctx.app_metadata.to_dict().items():
        print(f"  {key}: {value}")

    # ── Semantic ──────────────────────────────────────────────────────────────
    if any(m in selected for m in SEMANTIC_METHODS):
        print("\n  Running semantic methods...")
        all_results.update(run_semantic(patient, selected, config, ctx))

    # ── Set-based ─────────────────────────────────────────────────────────────
    if any(m in selected for m in SET_BASED_METHODS):
        print("\n  Running set-based methods...")
        all_results.update(run_set_based(patient, selected, config, ctx))

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    if "tfidf" in selected:
        print("\n  Running TF-IDF methods...")
        all_results.update(run_tfidf(patient, selected, config, ctx))

    # ── Transformer ───────────────────────────────────────────────────────────
    if "transformer" in selected:
        print("\n  Running transformer methods...")
        alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
        patient_dict = {
            "patient_id": patient.patient_id,
            "raw_text": patient.raw_text,
            "hpo_terms": sorted(patient.hpo_terms),
        }
        transformer_results = run_transformer(
            disease_profiles=ctx.disease_profiles,
            hpo_labels=hpo_labels,
            patient=patient_dict,
            alias_to_canonical=alias_to_canonical,
            top_k=config.top_k,
        )
        all_raw_results.update(transformer_results)

    # ── LLM ───────────────────────────────────────────────────────────────────
    if "llm" in selected:
        print("\n  Running LLM methods...")
        patient_dict = {
            "patient_id": patient.patient_id,
            "raw_text": patient.raw_text,
            "hpo_terms": sorted(patient.hpo_terms),
        }
        llm_results = run_llm(
            patient=patient_dict,
            hpo_labels=hpo_labels,
            disease_profiles=ctx.disease_profiles,
            top_k=config.top_k,
        )
        all_raw_results["llm"] = llm_results

    # ── Display ───────────────────────────────────────────────────────────────
    for method_name, results in all_results.items():
        print_results_table(method_name, results)

    for method_name, results in all_raw_results.items():
        print_raw_results(method_name, results)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(all_results, ctx.app_metadata.to_dict())


  # ── Cache — save all results together for later comparison ────────────────
    save_run_cache(
        patient_id=patient.patient_id,
        config=config,
        similarity_results=all_results,
        raw_results=all_raw_results,
        app_metadata=ctx.app_metadata.to_dict(),
    )


if __name__ == "__main__":
    main()
    