"""
Run phenotype extraction on medicalCases.json and save results.

Input : test_data/medicalCases.json
        Format: { "ORPHA_CODE": "clinical text", ... }

Output: outputs/evaluation/medical_cases/
        - extraction_results.json   — full extraction with provenance per case
        - test_cases.json           — [[hpo_terms, ground_truth], ...] for evaluator
        - extraction_summary.txt    — per-method stats

Usage:
    # Run with fast methods only (recommended for first run)
    python run_medical_cases_extraction.py --methods dictionary fast_hpo_cr

    # Test on first 10 cases with dictionary + fast_hpo_cr
    python run_medical_cases_extraction.py --methods dictionary fast_hpo_cr --limit 10

    # Run all methods for first 10 cases
    python run_medical_cases_extraction.py \
    --methods dictionary fast_hpo_cr chatgpt phenobrain_api \
    --limit 10

    # Run chatgpt method for first 5 cases - hallucinatios possible - from input text it produces hpo labels and then we map to hpo ids with hpo_labels json file.
    python run_medical_cases_extraction.py --methods chatgpt --limit 5

    # Resume interrupted run (skips already processed cases)
    python run_medical_cases_extraction.py --methods dictionary fast_hpo_cr --resume
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

INPUT_PATH  = PROJECT_ROOT / "test_data" / "medicalCases.json"
OUTPUT_DIR  = OUTPUTS_DIR / "evaluation" / "medical_cases"


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
        help="Skip cases already in extraction_results.json",
    )
    return parser.parse_args()


def load_existing_results(output_dir: Path) -> dict:
    path = output_dir / "extraction_results.json"
    if path.exists():
        return load_json(path)
    return {}


def print_summary(results: dict, methods: list[str], elapsed: float) -> None:
    n = len(results)
    method_counts = {m: 0 for m in methods}
    hpo_counts = []

    for case in results.values():
        hpo_terms = case.get("hpo_terms", [])
        hpo_counts.append(len(hpo_terms))
        for term in case.get("extracted_terms", []):
            method = term.get("method", "")
            for m in methods:
                if m in method:
                    method_counts[m] += 1

    print(f"\n{'=' * 60}")
    print(f"  Extraction Summary")
    print(f"{'=' * 60}")
    print(f"  Cases processed : {n}")
    print(f"  Total time      : {elapsed/60:.1f} min")
    print(f"  Avg time/case   : {elapsed/n:.1f}s" if n > 0 else "")
    print(f"  Avg HPO terms   : {sum(hpo_counts)/n:.1f}" if n > 0 else "")
    print(f"  Min HPO terms   : {min(hpo_counts) if hpo_counts else 0}")
    print(f"  Max HPO terms   : {max(hpo_counts) if hpo_counts else 0}")
    print(f"  Cases with 0 HPO: {sum(1 for c in hpo_counts if c == 0)}")
    print(f"\n  Terms found per method:")
    for method, count in method_counts.items():
        print(f"    {method:<20}: {count}")
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
    print(f"  Methods: {args.methods}\n")

    # Load HPO labels
    hpo_labels = load_json(HPO_LABELS_PATH)

    # Load existing results if resuming
    results = load_existing_results(OUTPUT_DIR) if args.resume else {}
    skipped = 0

    start_time = time.time()
    processed = 0
    failed = 0

    for i, orpha_code in enumerate(orpha_codes):
        case_id = f"medcase_{orpha_code}"

        # Skip if already processed
        if args.resume and case_id in results:
            skipped += 1
            continue

        raw_text = raw_cases[orpha_code]
        ground_truth = [f"ORPHA:{orpha_code}"]

        print(f"[{i+1:>5}/{total}] ORPHA:{orpha_code} | {len(raw_text)} chars")

        try:
            case_start = time.time()
            patient, extracted_terms = build_patient_profile(
                patient_id=case_id,
                raw_text=raw_text,
                hpo_labels=hpo_labels,
                methods=args.methods,
            )
            case_elapsed = time.time() - case_start

            results[case_id] = {
                "orpha_code": orpha_code,
                "ground_truth": ground_truth,
                "hpo_terms": patient["hpo_terms"],
                "extracted_terms": extracted_terms,
                "n_hpo_terms": len(patient["hpo_terms"]),
                "elapsed_seconds": round(case_elapsed, 2),
                "methods_used": args.methods,
            }
            processed += 1
            print(f"         ✓ {len(patient['hpo_terms'])} HPO terms in {case_elapsed:.1f}s")

        except Exception as e:
            failed += 1
            print(f"         ✗ ERROR: {e}")
            results[case_id] = {
                "orpha_code": orpha_code,
                "ground_truth": ground_truth,
                "hpo_terms": [],
                "extracted_terms": [],
                "n_hpo_terms": 0,
                "error": str(e),
                "methods_used": args.methods,
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
        if case["hpo_terms"]  # skip cases with no HPO terms extracted
    ]
    save_json(test_cases, OUTPUT_DIR / "test_cases.json")
    print(f"Saved test cases         -> {OUTPUT_DIR / 'test_cases.json'}")
    print(f"  {len(test_cases)} cases with HPO terms (out of {len(results)} total)")

    # Print and save summary
    print_summary(results, args.methods, elapsed)

    summary_lines = [
        f"Cases processed : {processed}",
        f"Cases skipped   : {skipped} (resumed)",
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
