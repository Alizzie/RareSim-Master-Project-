"""
RareSim Transformer Batch Runner

Runs all transformer models on every test case and caches results.
The disease embedding cache is built once and reused — ranking is
near-instant after the first run.

Requires GPU.

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage: (whichever gpu is free, change 5 to your gpu id)
    CUDA_VISIBLE_DEVICES=5 python evaluation/run_transformer.py \\
        --test-set test_data/test_cases/MME.json
"""

import argparse
import sys
import time
from pathlib import Path

from _batch_utils import (
    SRC_DIR, PROJECT_ROOT, CACHE_BASE_DIR,
    load_test_cases,
    cache_path_for, methods_already_cached, save_cache,
    print_header, print_case, print_case_ok, print_case_err, print_summary,
    add_common_args,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.context import AppContext
from shared.io import load_json
from shared.paths import HPO_LABELS_PATH, ALIAS_TO_CANONICAL_PATH
from core.schemas import PatientProfile


def run(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """Run all transformer models on every test case."""
    from similarity_methods.transformer.config import MODEL_LIST, CANDIDATE_POOL_SIZE
    from similarity_methods.transformer.retriever import DiseaseRetriever

    cache_dir = CACHE_BASE_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("transformer", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    hpo_labels = load_json(HPO_LABELS_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)

    print(f"Models: {MODEL_LIST}")
    print("Building / loading transformer embedding cache...")
    retriever = DiseaseRetriever(
        disease_profiles=ctx.disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=MODEL_LIST,
    )
    retriever.warmup(preload_models=True)
    print("  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, MODEL_LIST):
            skipped += 1
            continue

        print_case(index, total, hpo_terms, ground_truth)

        try:
            method_elapsed: dict[str, float] = {}
            all_results: dict = {}
            wall_start = time.time()

            patient_dict = {
                "patient_id": f"eval_case_{index:04d}",
                "raw_text": "",
                "hpo_terms": list(hpo_terms),
            }

            for model_name in MODEL_LIST:
                t0 = time.time()
                all_results[model_name] = retriever.rank(
                    model_name=model_name,
                    patient=patient_dict,
                    top_k=top_k,
                    candidate_pool_size=CANDIDATE_POOL_SIZE,
                )
                method_elapsed[model_name] = round(time.time() - t0, 3)

            elapsed = time.time() - wall_start
            total_time += elapsed

            save_cache(
                cache_file, index, hpo_terms, ground_truth,
                all_results, method_elapsed, elapsed,
            )
            processed += 1
            print_case_ok(elapsed, total_time, processed, total - index - 1)

        except Exception as e:
            failed += 1
            print_case_err(e)
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim transformer batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        test_set_path=args.test_set,
        resume=not args.no_resume,
        limit=args.limit,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
