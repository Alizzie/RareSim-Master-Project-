"""
RareSim HPO2Vec Batch Runner

Runs the HPO2Vec pipeline on every test case and caches results.
Loads the pre-trained Word2Vec model from outputs/hpo2vec/hpo2vec_model.

CPU only — fast after the model is loaded.

Cache:
    results/evaluation/{test_set_name}/cache/case_NNNN.json

Usage:
    python evaluation/run_hpo2vec.py \\
        --test-set test_data/test_cases/MME.json

Prerequisites:
    Train the model first if it does not exist:
        python pipelines/hpo2vec_pipeline.py
"""

import argparse
import sys
import time
from pathlib import Path

from _batch_utils import (
    PROJECT_ROOT, SRC_DIR, CACHE_BASE_DIR,
    load_test_cases, build_patient,
    cache_path_for, methods_already_cached, save_cache,
    print_header, print_case, print_case_ok, print_case_err, print_summary,
    add_common_args,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.context import AppContext
from shared.io import load_json
from shared.math import preprocess_ancestor_sets
from shared.paths import ALIAS_TO_CANONICAL_PATH
from core.schemas import PatientProfile

METHOD_NAME = "hpo2vec"


def run(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """Run HPO2Vec on every test case."""
    from gensim.models import Word2Vec

    pipelines_dir = str(PROJECT_ROOT / "pipelines")
    if pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)
    from hpo2vec_pipeline import rank_diseases

    cache_dir = CACHE_BASE_DIR / test_set_path.stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header(METHOD_NAME, test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    model_path = PROJECT_ROOT / "outputs" / "hpo2vec" / "hpo2vec_model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"HPO2Vec model not found at {model_path}.\n"
            "Train it first: python pipelines/hpo2vec_pipeline.py"
        )

    print(f"Loading HPO2Vec model from {model_path}...")
    model = Word2Vec.load(str(model_path))
    print(f"  Vocabulary size  : {len(model.wv)} nodes")

    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
    ic_values = load_json(PROJECT_ROOT / "outputs" / "shared" / "information_content.json")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print("  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and methods_already_cached(cache_file, [METHOD_NAME]):
            skipped += 1
            continue

        print_case(index, total, hpo_terms, ground_truth)

        try:
            patient = build_patient(index, hpo_terms, ancestor_sets)
            patient_dict = {
                "patient_id": patient.patient_id,
                "raw_text": "",
                "hpo_terms": list(patient.hpo_terms),
                "propagated_hpo_terms": list(patient.propagated_hpo_terms),
            }

            t0 = time.time()
            rankings = rank_diseases(
                disease_profiles=ctx.disease_profiles,
                patient=patient_dict,
                model=model,
                ic_values=ic_values,
                alias_to_canonical=alias_to_canonical,
                use_propagated=True,
                top_k=top_k,
            )
            elapsed = round(time.time() - t0, 3)
            total_time += elapsed

            save_cache(
                cache_file, index, hpo_terms, ground_truth,
                {METHOD_NAME: rankings}, {METHOD_NAME: elapsed}, elapsed,
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
        description="RareSim HPO2Vec batch runner",
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
