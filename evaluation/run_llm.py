"""
RareSim LLM Batch Runner

Runs all configured LLM models on every test case and caches results.
Each model is loaded once, run over all cases, then unloaded before
the next model is loaded.

Requires GPU.

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage: (whichever gpu is free, change 5 to your gpu id)
    CUDA_VISIBLE_DEVICES=5 python evaluation/run_llm.py \\
        --test-set test_data/test_cases/MME.json
"""

import argparse
import sys
import time
from pathlib import Path

from _batch_utils import (
    SRC_DIR, CACHE_BASE_DIR,
    load_test_cases,
    cache_path_for, methods_already_cached, save_cache,
    print_header, print_case, print_case_ok, print_case_err, print_summary,
    add_common_args,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.context import AppContext
from shared.io import load_json
from shared.paths import HPO_LABELS_PATH
from core.schemas import PatientProfile


def run(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """Run all LLM models on every test case."""
    from similarity_methods.llm.methods import (
        unload_pipeline, load_hf_pipeline,
        build_retrieval_prompt, query_hf, parse_retrieval_output,
    )
    from similarity_methods.llm.config import LLM_MODEL_LIST, MAX_NEW_TOKENS_RETRIEVAL

    cache_dir = CACHE_BASE_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("llm", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)

    print(f"Loaded {total} test cases.")
    print(f"Models  : {LLM_MODEL_LIST}")
    print(f"Warning : LLM is slow (~3 min/case). Est. total: {total * len(LLM_MODEL_LIST) * 3} min\n")

    hpo_labels = load_json(HPO_LABELS_PATH)
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for model_name in LLM_MODEL_LIST:
        print(f"\n[llm] Loading model: {model_name}")
        pipe = load_hf_pipeline(model_name, MAX_NEW_TOKENS_RETRIEVAL)

        for index, (hpo_terms, ground_truth) in enumerate(cases):
            cache_file = cache_path_for(cache_dir, index)

            if resume and methods_already_cached(cache_file, [model_name]):
                skipped += 1
                continue

            print_case(index, total, hpo_terms, ground_truth)

            try:
                patient_dict = {
                    "patient_id": f"eval_case_{index:04d}",
                    "raw_text": "",
                    "hpo_terms": list(hpo_terms),
                }

                t0 = time.time()
                prompt = build_retrieval_prompt(patient_dict, hpo_labels, top_k)
                generated = query_hf(prompt, pipe, max_tokens=MAX_NEW_TOKENS_RETRIEVAL)
                model_results = parse_retrieval_output(
                    generated, ctx.disease_profiles, model_name, top_k
                )
                elapsed = round(time.time() - t0, 3)
                total_time += elapsed

                save_cache(
                    cache_file, index, hpo_terms, ground_truth,
                    {model_name: model_results}, {model_name: elapsed}, elapsed,
                )
                processed += 1
                print_case_ok(elapsed, total_time, processed, total - index - 1)

            except Exception as e:
                failed += 1
                print_case_err(e)
                (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

        unload_pipeline(pipe)

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim LLM batch runner",
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
