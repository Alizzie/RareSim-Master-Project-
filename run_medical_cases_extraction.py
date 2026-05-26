"""
Run phenotype extraction on medicalCases.json and save results.

Input : test_data/medicalCases.json
        Format: { "ORPHA_CODE": "clinical text", ... }

Output: outputs/evaluation/medical_cases/
        - extraction_results.json   — full extraction with provenance per case
        - test_cases.json           — [[hpo_terms, ground_truth], ...] for evaluator
        - extraction_summary.txt    — per-method stats

Usage:
    # Run with fast methods only
    python run_medical_cases_extraction.py --methods dictionary fast_hpo_cr

    # Test on first 10 cases with dictionary + fast_hpo_cr
    python run_medical_cases_extraction.py --methods dictionary fast_hpo_cr --limit 10

    # Run all methods for first 10 cases
    python run_medical_cases_extraction.py \
    --methods dictionary biomedical_ner fast_hpo_cr chatgpt phenobrain_api \
    --limit 10

    # Run chatgpt method for first 5 cases - hallucinations possible - from input
    # text it produces hpo labels and then we map to hpo ids with hpo_labels json.
    python run_medical_cases_extraction.py --methods chatgpt --limit 5

"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.io import load_json, save_json
from shared.paths import HPO_LABELS_PATH, OUTPUTS_DIR
from shared.phenotype import build_patient_profile

INPUT_PATH = PROJECT_ROOT / "test_data" / "medicalCases.json"
OUTPUT_DIR = OUTPUTS_DIR / "evaluation" / "medical_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phenotype extraction on medicalCases.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help=f"Path to medicalCases.json (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dictionary", "fast_hpo_cr"],
        choices=["dictionary", "biomedical_ner", "fast_hpo_cr", "chatgpt", "phenobrain_api"],
        help="Extraction methods to use (default: dictionary fast_hpo_cr)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N cases (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Merge new method results into existing cases, skip if all methods already run",
    )
    return parser.parse_args()


def load_existing_results(output_dir: Path) -> dict:
    path = output_dir / "extraction_results.json"
    if path.exists():
        return load_json(path)
    return {}


def merge_case_results(existing: dict, new_hpo_terms: list, new_extracted_terms: list, new_methods: list) -> dict:
    """Merge new extraction results into an existing case, combining HPO terms and provenance."""
    # Merge HPO terms
    merged_hpo = sorted(set(existing.get("hpo_terms", [])) | set(new_hpo_terms))

    # Merge extracted terms — deduplicate by (hpo_id, method)
    existing_terms = {
        (t["hpo_id"], t["method"]): t
        for t in existing.get("extracted_terms", [])
    }
    for t in new_extracted_terms:
        existing_terms[(t["hpo_id"], t["method"])] = t

    # Merge methods_used
    merged_methods = sorted(set(existing.get("methods_used", [])) | set(new_methods))

    return {
        **existing,
        "hpo_terms": merged_hpo,
        "extracted_terms": list(existing_terms.values()),
        "n_hpo_terms": len(merged_hpo),
        "methods_used": merged_methods,
    }


def print_summary(results: dict, methods: list, elapsed: float) -> None:
    n = len(results)
    if n == 0:
        return

    hpo_counts = [case.get("n_hpo_terms", 0) for case in results.values()]

    # Count terms per method across all cases
    method_counts: dict[str, int] = {}
    for case in results.values():
        for t in case.get("extracted_terms", []):
            m = t.get("method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"  Extraction Summary")
    print(f"{'=' * 60}")
    print(f"  Cases processed : {n}")
    print(f"  Total time      : {elapsed/60:.1f} min")
    print(f"  Avg time/case   : {elapsed/n:.1f}s")
    print(f"  Avg HPO terms   : {sum(hpo_counts)/n:.1f}")
    print(f"  Min HPO terms   : {min(hpo_counts)}")
    print(f"  Max HPO terms   : {max(hpo_counts)}")
    print(f"  Cases with 0 HPO: {sum(1 for c in hpo_counts if c == 0)}")
    print(f"\n  Terms found per method (all cases):")
    for method, count in sorted(method_counts.items()):
        print(f"    {method:<35}: {count}")
    print(f"{'=' * 60}\n")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load input
    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}")
        print(f"  Place medicalCases.json at: {INPUT_PATH}")
        sys.exit(1)

    print(f"Loading {args.input.name}...")
    raw_cases = load_json(args.input)
    orpha_codes = list(raw_cases.keys())
    if args.limit:
        orpha_codes = orpha_codes[:args.limit]
    total = len(orpha_codes)
    print(f"  {total} cases to process")
    print(f"  Methods: {args.methods}")
    print(f"  Resume : {args.resume}\n")

    # Load HPO labels
    hpo_labels = load_json(HPO_LABELS_PATH)

    # Load existing results if resuming
    results = load_existing_results(OUTPUT_DIR) if args.resume else {}

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for i, orpha_code in enumerate(orpha_codes):
        case_id = f"medcase_{orpha_code}"
        raw_text = raw_cases[orpha_code]
        ground_truth = [f"ORPHA:{orpha_code}"]

        # Skip if all requested methods already run for this case
        if args.resume and case_id in results:
            already_run = set(results[case_id].get("methods_used", []))
            remaining = [m for m in args.methods if m not in already_run]
            if not remaining:
                skipped += 1
                continue
            # Only run the remaining methods
            run_methods = remaining
            print(f"[{i+1:>5}/{total}] ORPHA:{orpha_code} | merging methods: {run_methods}")
        else:
            run_methods = args.methods
            print(f"[{i+1:>5}/{total}] ORPHA:{orpha_code} | {len(raw_text)} chars")

        try:
            case_start = time.time()
            patient, extracted_terms = build_patient_profile(
                patient_id=case_id,
                raw_text=raw_text,
                hpo_labels=hpo_labels,
                methods=run_methods,
            )
            case_elapsed = time.time() - case_start

            if case_id in results:
                # Merge into existing case
                results[case_id] = merge_case_results(
                    existing=results[case_id],
                    new_hpo_terms=patient["hpo_terms"],
                    new_extracted_terms=extracted_terms,
                    new_methods=run_methods,
                )
            else:
                # New case
                results[case_id] = {
                    "orpha_code": orpha_code,
                    "ground_truth": ground_truth,
                    "hpo_terms": patient["hpo_terms"],
                    "extracted_terms": extracted_terms,
                    "n_hpo_terms": len(patient["hpo_terms"]),
                    "elapsed_seconds": round(case_elapsed, 2),
                    "methods_used": run_methods,
                }

            processed += 1
            print(f"         ✓ {results[case_id]['n_hpo_terms']} HPO terms in {case_elapsed:.1f}s")

        except Exception as e:
            failed += 1
            print(f"         ✗ ERROR: {e}")
            if case_id not in results:
                results[case_id] = {
                    "orpha_code": orpha_code,
                    "ground_truth": ground_truth,
                    "hpo_terms": [],
                    "extracted_terms": [],
                    "n_hpo_terms": 0,
                    "error": str(e),
                    "methods_used": run_methods,
                }

        # Save incrementally every 50 cases
        if (i + 1) % 50 == 0:
            save_json(results, OUTPUT_DIR / "extraction_results.json")
            print(f"  [checkpoint] Saved {len(results)} cases")

    elapsed = time.time() - start_time

    # Save final results
    save_json(results, OUTPUT_DIR / "extraction_results.json")
    print(f"\nSaved extraction results -> {OUTPUT_DIR / 'extraction_results.json'}")

    # Save as test_cases.json for evaluator: [[hpo_terms, ground_truth], ...]
    test_cases = [
        [case["hpo_terms"], case["ground_truth"]]
        for case in results.values()
        if case["hpo_terms"]
    ]
    save_json(test_cases, OUTPUT_DIR / "test_cases.json")
    print(f"Saved test cases         -> {OUTPUT_DIR / 'test_cases.json'}")
    print(f"  {len(test_cases)} cases with HPO terms (out of {len(results)} total)")

    # Print and save summary
    print_summary(results, args.methods, elapsed)

    summary_lines = [
        f"Cases processed : {processed}",
        f"Cases skipped   : {skipped} (all methods already run)",
        f"Cases failed    : {failed}",
        f"Cases with HPO  : {len(test_cases)}",
        f"Methods         : {args.methods}",
        f"Total time      : {elapsed/60:.1f} min",
    ]
    summary_path = OUTPUT_DIR / "extraction_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"Saved summary            -> {summary_path}")


if __name__ == "__main__":
    main()
