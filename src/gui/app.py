"""
Terminal interface for the rare disease similarity pipeline.

Two input modes:
    --text   Raw clinical text → extract HPO terms → run similarity
    --hpo    Pre-extracted HPO terms (list or JSON file) → run similarity directly

Usage:
    # From raw clinical text (extraction + similarity):
    python src/gui/app.py --text "Patient with cerebellar ataxia and macrocephaly."

    # From raw text with specific extraction methods:
    python src/gui/app.py \\
        --text "Patient with cerebellar ataxia and macrocephaly." \\
        --extraction-methods fast_hpo_cr chatgpt

    # From pre-extracted HPO terms (comma-separated):
    python src/gui/app.py --hpo HP:0001251,HP:0000256

    # From a patient JSON file:
    python src/gui/app.py --patient outputs/shared/example_patient.json

    # Use example patient, all methods:
    python src/gui/app.py --defaults

    # Select specific similarity methods:
    python src/gui/app.py --text "..." --methods semantic_resnik_bma tfidf

    # Change top-k:
    python src/gui/app.py --defaults --top-k 5
"""

import argparse
from pathlib import Path

from shared.cache import save_run_cache
from shared.context import AppContext
from shared.io import load_json, load_patient_with_extraction, save_json
from shared.paths import (
    ALIAS_TO_CANONICAL_PATH,
    HPO_LABELS_PATH,
    SHARED_DIR,
)
from shared.pipeline import PipelineConfig
from hpo_extraction import build_patient_profile
from gui.utils import (
    GUI_DIR,
    check_artifacts_exist,
    print_app_metadata,
    print_results_table,
    print_raw_results,
    save_results,
    prompt_patient,
    prompt_methods,
)
from gui.summary import (
    build_disease_summary,
    build_timing_summary,
    print_disease_summary,
    print_timing_summary,
)

from shared.result import MethodResults
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

EXTRACTION_METHODS = [
    "dictionary",
    "biomedical_ner",
    "fast_hpo_cr",
    "chatgpt",
    "phenobrain_api",
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


def main() -> None:
    """Main function to run the terminal interface."""
    check_artifacts_exist()
    args = parse_args()

    hpo_labels = load_json(HPO_LABELS_PATH)

    print("=" * 64)
    print("  Rare Disease Similarity Pipeline")
    print("=" * 64)

    # ── Input mode ────────────────────────────────────────────────────────────

    if args.text:
        # Mode 1: raw clinical text → extract HPO terms → similarity
        print(f"\nInput mode : raw text")
        print(f"Extraction : {args.extraction_methods}")
        print(f"\nExtracting HPO terms...")

        patient_dict, extracted_terms = build_patient_profile(
            patient_id="text_input_patient",
            raw_text=args.text,
            hpo_labels=hpo_labels,
            methods=args.extraction_methods,
        )

        print(f"\nExtracted {len(patient_dict['hpo_terms'])} HPO terms:")
        for t in extracted_terms:
            print(f"  {t['hpo_id']} | {t['label']} | method={t['method']}")

        if not patient_dict["hpo_terms"]:
            print("\n[warning] No HPO terms extracted — check your text or try different extraction methods.")
            return

        # Save to temp file for load_patient_with_extraction
        tmp_path = SHARED_DIR / "extracted_patient.json"
        save_json(patient_dict, tmp_path)
        patient = load_patient_with_extraction(tmp_path, hpo_labels)

    elif args.hpo:
        hpo_terms = [t.strip() for t in args.hpo.split(",") if t.strip()]
        print(f"\nInput mode : HPO terms")
        print(f"HPO terms  : {hpo_terms}")

        # Build a minimal patient profile with propagation
        from shared.math import preprocess_ancestor_sets, get_ancestors_inclusive
        from shared.paths import HPO_ANCESTORS_PATH
        ancestors = load_json(HPO_ANCESTORS_PATH)
        ancestor_sets = preprocess_ancestor_sets(ancestors)
        propagated = set()
        for term in hpo_terms:
            propagated |= get_ancestors_inclusive(term, ancestor_sets)

        patient_dict = {
            "patient_id": "hpo_input_patient",
            "raw_text": "",
            "hpo_terms": hpo_terms,
            "propagated_hpo_terms": sorted(propagated),
            "methods_used": ["direct_input"],
        }
        tmp_path = SHARED_DIR / "hpo_input_patient.json"
        save_json(patient_dict, tmp_path)
        patient = load_patient_with_extraction(tmp_path, hpo_labels)

    elif args.patient:
        # Mode 3: patient JSON file
        print(f"\nInput mode : patient file")
        patient = load_patient_with_extraction(args.patient, hpo_labels)
        print(f"Patient    : {args.patient.name}")

    elif args.defaults:
        # Mode 4: example patient
        print(f"\nInput mode : default example patient")
        patient = load_patient_with_extraction(DEFAULTS["patient_path"], hpo_labels)

    else:
        # Mode 5: interactive prompt
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
    print("\nLoading shared context...")
    ctx = AppContext.load(patient, config.use_canonical_profiles)
    print_app_metadata(ctx.app_metadata)

    # -─ Run pipelines ─────────────────────────────────────────────────────────────
    print("\nRunning pipeline...")
    all_results: dict[str, MethodResults] = {}

    # for methods that don't fit SimilarityResult format
    all_raw_results: dict[str, list[dict]] = {}

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
        all_raw_results.update(
            run_transformer(
                disease_profiles=ctx.disease_profiles,
                hpo_labels=hpo_labels,
                patient=patient_dict,
                alias_to_canonical=alias_to_canonical,
                top_k=config.top_k,
            )
        )

    # ── LLM ───────────────────────────────────────────────────────────────────
    if "llm" in selected:
        print("\n  Running LLM methods...")
        patient_dict = {
            "patient_id": patient.patient_id,
            "raw_text": patient.raw_text,
            "hpo_terms": sorted(patient.hpo_terms),
        }
        all_raw_results["llm"] = run_llm(
            patient=patient_dict,
            hpo_labels=hpo_labels,
            disease_profiles=ctx.disease_profiles,
            top_k=config.top_k,
        )

    # ── Display ───────────────────────────────────────────────────────────────
    for method_results in all_results.values():
        print_results_table(method_results)

    for method_name, results in all_raw_results.items():
        try:
            print_raw_results(method_name, results)
        except Exception as e:
            print(f"  [warning] Could not print results for {method_name}: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(all_results, ctx.app_metadata)

    # ── Cache — save all results together for later comparison ────────────────
    save_run_cache(
        patient_id=patient.patient_id,
        config=config,
        similarity_results=all_results,
        raw_results=all_raw_results,
        app_metadata=ctx.app_metadata,
    )

    # ── Summaries ───────────────────────────────────────────────────────────────
    disease_summary = build_disease_summary(all_results)
    timing_summary = build_timing_summary(all_results)
    print_disease_summary(disease_summary)
    print_timing_summary(timing_summary)

    save_json(disease_summary, GUI_DIR / "disease_summary.json")
    save_json(timing_summary, GUI_DIR / "timing_summary.json")


if __name__ == "__main__":
    main()
