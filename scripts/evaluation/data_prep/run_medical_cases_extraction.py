"""
Run phenotype extraction on medicalCases.json and save results.

Input : data/datasets/free_text/medicalCases.json
        Format: { "ORPHA_CODE": "clinical text", ... }

Output: data/datasets/free_text/extracted_medical_cases/
        - extraction_results.json   — full extraction with provenance per case
        - test_medical_cases.json   — [[hpo_terms, ground_truth], ...] for evaluator
        - extraction_summary.txt    — per-method stats

Usage:
    # Run with fast methods only
    python scripts/evaluation/data_prep/run_medical_cases_extraction.py \
    --methods dictionary fast_hpo_cr

    # Test on first 200 cases with dictionary + fast_hpo_cr
    python scripts/evaluation/data_prep/run_medical_cases_extraction.py \
        --methods dictionary fast_hpo_cr \
        --limit 200

    # Run all methods for first 200 cases
    python scripts/evaluation/data_prep/run_medical_cases_extraction.py \
        --methods dictionary biomedical_ner fast_hpo_cr chatgpt phenobrain_api \
        --limit 200

    # Run chatgpt method for first 5 cases - hallucinations possible - from input
    # text it produces hpo labels and then we map to hpo ids with hpo_labels json.
    python scripts/evaluation/data_prep/run_medical_cases_extraction.py --methods chatgpt --limit 5
"""
# pylint: disable=broad-exception-caught,too-many-locals,too-many-statements
import argparse
import sys
import time
from pathlib import Path
from typing import Any, cast

from raresim.hpo_extraction import build_patient_profile
from raresim.utils.io import load_json, save_json
from raresim.utils.paths import EXTRACTED_MEDICAL_CASES_DIR, HPO_LABELS_PATH, MEDICAL_CASES_DIR

INPUT_PATH = MEDICAL_CASES_DIR / "medicalCases.json"
OUTPUT_DIR = EXTRACTED_MEDICAL_CASES_DIR

JsonDict = dict[str, Any]
ExtractedTerm = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        choices=[
            "dictionary",
            "biomedical_ner",
            "fast_hpo_cr",
            "chatgpt",
            "phenobrain_api",
        ],
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


def load_existing_results(output_dir: Path) -> dict[str, JsonDict]:
    """Load previous extraction results if they exist."""
    path = output_dir / "extraction_results.json"
    if path.exists():
        return cast(dict[str, JsonDict], load_json(path))
    return {}


def merge_case_results(
    existing: JsonDict,
    new_hpo_terms: list[str],
    new_extracted_terms: list[ExtractedTerm],
    new_methods: list[str],
) -> JsonDict:
    """Merge new extraction results into an existing case."""
    existing_hpo_terms = cast(list[str], existing.get("hpo_terms", []))
    merged_hpo = sorted(set(existing_hpo_terms) | set(new_hpo_terms))

    existing_extracted_terms = cast(
        list[ExtractedTerm], existing.get("extracted_terms", [])
    )
    existing_terms = {
        (str(term["hpo_id"]), str(term["method"])): term
        for term in existing_extracted_terms
    }

    for term in new_extracted_terms:
        existing_terms[(str(term["hpo_id"]), str(term["method"]))] = term

    existing_methods = cast(list[str], existing.get("methods_used", []))
    merged_methods = sorted(set(existing_methods) | set(new_methods))

    return {
        **existing,
        "hpo_terms": merged_hpo,
        "extracted_terms": list(existing_terms.values()),
        "n_hpo_terms": len(merged_hpo),
        "methods_used": merged_methods,
    }


def print_summary(
    results: dict[str, JsonDict],
    elapsed: float,
) -> None:
    """Print extraction summary statistics."""
    n_cases = len(results)
    if n_cases == 0:
        return

    hpo_counts = [
        int(case.get("n_hpo_terms", 0))
        for case in results.values()
    ]

    method_counts: dict[str, int] = {}
    for case in results.values():
        extracted_terms = cast(
            list[ExtractedTerm], case.get("extracted_terms", [])
        )
        for term in extracted_terms:
            method = str(term.get("method", "unknown"))
            method_counts[method] = method_counts.get(method, 0) + 1

    print(f"\n{'=' * 60}")
    print("  Extraction Summary")
    print(f"{'=' * 60}")
    print(f"  Cases processed : {n_cases}")
    print(f"  Total time      : {elapsed / 60:.1f} min")
    print(f"  Avg time/case   : {elapsed / n_cases:.1f}s")
    print(f"  Avg HPO terms   : {sum(hpo_counts) / n_cases:.1f}")
    print(f"  Min HPO terms   : {min(hpo_counts)}")
    print(f"  Max HPO terms   : {max(hpo_counts)}")
    print(f"  Cases with 0 HPO: {sum(1 for count in hpo_counts if count == 0)}")
    print("\n  Terms found per method (all cases):")
    for method, count in sorted(method_counts.items()):
        print(f"    {method:<35}: {count}")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Run phenotype extraction on medical cases."""
    args = parse_args()
    methods = cast(list[str], args.methods)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"[error] Input file not found: {args.input}")
        print(f"  Place medicalCases.json at: {INPUT_PATH}")
        sys.exit(1)

    print(f"Loading {args.input.name}...")
    raw_cases = cast(dict[str, str], load_json(args.input))

    orpha_codes = list(raw_cases.keys())
    if args.limit is not None:
        orpha_codes = orpha_codes[: args.limit]

    total = len(orpha_codes)
    print(f"  {total} cases to process")
    print(f"  Methods: {methods}")
    print(f"  Resume : {args.resume}\n")

    hpo_labels = cast(dict[str, str], load_json(HPO_LABELS_PATH))

    results: dict[str, JsonDict] = (
        load_existing_results(OUTPUT_DIR) if args.resume else {}
    )

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for i, orpha_code in enumerate(orpha_codes):
        case_id = f"medcase_{orpha_code}"
        raw_text = raw_cases[orpha_code]
        ground_truth = [f"ORPHA:{orpha_code}"]

        if args.resume and case_id in results:
            already_run = set(
                cast(list[str], results[case_id].get("methods_used", []))
            )
            remaining = [method for method in methods if method not in already_run]

            if not remaining:
                skipped += 1
                continue

            run_methods = remaining
            print(
                f"[{i + 1:>5}/{total}] ORPHA:{orpha_code} | "
                f"merging methods: {run_methods}"
            )
        else:
            run_methods = methods
            print(f"[{i + 1:>5}/{total}] ORPHA:{orpha_code} | {len(raw_text)} chars")

        try:
            case_start = time.time()
            patient, extracted_terms_raw = build_patient_profile(
                patient_id=case_id,
                raw_text=raw_text,
                hpo_labels=hpo_labels,
                methods=run_methods,
            )

            hpo_terms = cast(list[str], patient["hpo_terms"])
            extracted_terms = cast(list[ExtractedTerm], extracted_terms_raw)
            case_elapsed = time.time() - case_start

            if case_id in results:
                results[case_id] = merge_case_results(
                    existing=results[case_id],
                    new_hpo_terms=hpo_terms,
                    new_extracted_terms=extracted_terms,
                    new_methods=run_methods,
                )
            else:
                results[case_id] = {
                    "orpha_code": orpha_code,
                    "ground_truth": ground_truth,
                    "hpo_terms": hpo_terms,
                    "extracted_terms": extracted_terms,
                    "n_hpo_terms": len(hpo_terms),
                    "elapsed_seconds": round(case_elapsed, 2),
                    "methods_used": run_methods,
                }

            processed += 1
            print(
                f"         ✓ {results[case_id]['n_hpo_terms']} "
                f"HPO terms in {case_elapsed:.1f}s"
            )

        except Exception as exc:  # pylint: disable=broad-exception-caught
            failed += 1
            print(f"         ✗ ERROR: {exc}")

            if case_id not in results:
                results[case_id] = {
                    "orpha_code": orpha_code,
                    "ground_truth": ground_truth,
                    "hpo_terms": [],
                    "extracted_terms": [],
                    "n_hpo_terms": 0,
                    "error": str(exc),
                    "methods_used": run_methods,
                }

        if (i + 1) % 50 == 0:
            save_json(results, OUTPUT_DIR / "extraction_results.json")
            print(f"  [checkpoint] Saved {len(results)} cases")

    elapsed = time.time() - start_time

    save_json(results, OUTPUT_DIR / "extraction_results.json")
    print(f"\nSaved extraction results -> {OUTPUT_DIR / 'extraction_results.json'}")

    test_cases = [
        [
            cast(list[str], case["hpo_terms"]),
            cast(list[str], case["ground_truth"]),
        ]
        for case in results.values()
        if cast(list[str], case["hpo_terms"])
    ]

    save_json(test_cases, OUTPUT_DIR / "test_medical_cases.json")
    print(f"Saved test cases         -> {OUTPUT_DIR / 'test_medical_cases.json'}")
    print(f"  {len(test_cases)} cases with HPO terms (out of {len(results)} total)")

    print_summary(results, elapsed)


if __name__ == "__main__":
    main()
