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

from _constants import ALL_METHODS, DEFAULTS, SEMANTIC_METHODS, SET_BASED_METHODS
from _cli_parser import parse_args
import _utils as gu
import _summary as gsum

from raresim.utils.math import preprocess_ancestor_sets, get_ancestors_inclusive
from raresim.utils.paths import HPO_ANCESTORS_PATH, GUI_DIR
from raresim.core.cache import save_run_cache
from raresim.core.context import AppContext
from raresim.utils.io import load_json, save_json
from raresim.utils.patient_loader import load_patient_with_extraction
from raresim.utils.paths import ALIAS_TO_CANONICAL_PATH, HPO_LABELS_PATH
from raresim.core.pipeline import PipelineConfig
from raresim.hpo_extraction import build_patient_profile

from raresim.types.result import MethodResults
from raresim.similarity_methods.semantic.pipeline import run as run_semantic
from raresim.similarity_methods.set_based.pipeline import run as run_set_based
from raresim.similarity_methods.tfidf.pipeline import run as run_tfidf
from raresim.similarity_methods.transformer.pipeline import run as run_transformer
from raresim.similarity_methods.llm.pipeline import run as run_llm


def main() -> None:
    """Main function to run the terminal interface."""
    gu.check_artifacts_exist()
    args = parse_args()

    hpo_labels = load_json(HPO_LABELS_PATH)

    print("=" * 64)
    print("  Rare Disease Similarity Pipeline")
    print("=" * 64)

    # ── Input mode ────────────────────────────────────────────────────────────

    if args.text:
        # Mode 1: raw clinical text → extract HPO terms → similarity
        print("\nInput mode : raw text")
        print(f"Extraction : {args.extraction_methods}")
        print("\nExtracting HPO terms...")

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
            print(
                "\n[warning] No HPO terms extracted — check your text or try different extraction methods."
            )
            return

        # Save to temp file for load_patient_with_extraction
        tmp_path = GUI_DIR / "extracted_patient.json"
        save_json(patient_dict, tmp_path)
        patient = load_patient_with_extraction(tmp_path, hpo_labels)

    elif args.hpo:
        hpo_terms = [t.strip() for t in args.hpo.split(",") if t.strip()]
        print(f"\nInput mode : HPO terms")
        print(f"HPO terms  : {hpo_terms}")

        # Build a minimal patient profile with propagation
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
        tmp_path = GUI_DIR / "hpo_input_patient.json"
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
        patient = gu.prompt_patient(DEFAULTS, hpo_labels)

    print(f"  HPO terms: {len(patient.hpo_terms)}")
    print(f"  Propagated terms: {len(patient.propagated_hpo_terms)}")

    # ── Methods ───────────────────────────────────────────────────────────────
    if args.methods:
        selected = args.methods
    elif args.defaults:
        selected = ALL_METHODS
    else:
        selected = gu.prompt_methods(ALL_METHODS)

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
    gu.print_app_metadata(ctx.app_metadata)

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
        gu.print_results_table(method_results)

    for method_name, results in all_raw_results.items():
        try:
            gu.print_raw_results(method_name, results)
        except Exception as e:
            print(f"  [warning] Could not print results for {method_name}: {e}")

    # ── Save ──────────────────────────────────────────────────────────────────
    gu.save_results(all_results, ctx.app_metadata)

    # ── Cache — save all results together for later comparison ────────────────
    save_run_cache(
        patient_id=patient.patient_id,
        config=config,
        similarity_results=all_results,
        raw_results=all_raw_results,
        app_metadata=ctx.app_metadata,
    )

    # ── Summaries ───────────────────────────────────────────────────────────────
    disease_summary = gsum.build_disease_summary(all_results)
    timing_summary = gsum.build_timing_summary(all_results)
    gsum.print_disease_summary(disease_summary)
    gsum.print_timing_summary(timing_summary)

    save_json(disease_summary, GUI_DIR / "disease_summary.json")
    save_json(timing_summary, GUI_DIR / "timing_summary.json")


if __name__ == "__main__":
    main()
