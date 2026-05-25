"""
File for running similarity pipelines on test cases and caching results.
Test cases source:
    https://zenodo.org/records/10774650
    Paper: https://www.nature.com/articles/s41746-025-01452-1

Runs similarity pipelines on test cases and caches results to disk.
Each pipeline is run separately and cached in its own directory.

Pipelines:
    cpu         → semantic/set/tfidf (CPU, ~2-4 hours for 88 cases)
    transformer → GPU required, fast after cache build (~0.05s per case)
    llm         → GPU required, slow - even with gpu, running 88 cases took 2.5 days in anna but it runs in the backgound with nohup.
    hpo2vec     → CPU, fast after model is loaded (~0.1s per case)

Cache locations:
    outputs/evaluation/cache/{test_set_name}/             ← cpu
    outputs/evaluation/cache/{test_set_name}_transformer/ ← transformer
    outputs/evaluation/cache/{test_set_name}_llm/         ← llm
    outputs/evaluation/cache/{test_set_name}_hpo2vec/     ← hpo2vec


Usage:
Step 1:

    # To run in background semantic/set/tfidf (CPU)
    nohup python src/evaluation/run_test_files.py \
        --test-set test_data/test_cases/MME.json > run_log.txt 2>&1 &
Step 2:

    # Run transformer (GPU) — make sure to set CUDA_VISIBLE_DEVICES to select GPU and change CUDA_VISIBLE_DEVICES=4 to the appropriate GPU index
    CUDA_VISIBLE_DEVICES=5 python src/evaluation/run_test_files.py \
        --test-set test_data/test_cases/MME.json --pipeline transformer
Step 3:

    # Run LLM (GPU) — make sure to set CUDA_VISIBLE_DEVICES to select GPU and change CUDA_VISIBLE_DEVICES=4 to the appropriate GPU index
    CUDA_VISIBLE_DEVICES=5 python src/evaluation/run_test_files.py \
        --test-set test_data/test_cases/MME.json --pipeline llm

Step 4:

    # Run HPO2Vec (CPU) — can run in parallel with transformer/llm
    nohup python src/evaluation/run_test_files.py \
        --test-set test_data/test_cases/MME.json --pipeline hpo2vec > run_log_hpo2vec.txt 2>&1 &

note: cpu and hpo2vec can be run in parallel with transformer and llm since they don't require GPU.
Transformer and llm should not be run at the same time on the same GPU to avoid memory issues.
for testing, use --limit to only run on the first N cases, e.g. --limit 10
example:
CUDA_VISIBLE_DEVICES=4 nohup python src/evaluation/run_test_files.py \
  --test-set test_data/test_cases/MME.json --pipeline llm --limit 10 \
  > outputs/evaluation/llm_log.txt 2>&1 &
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.schemas import PatientProfile
from shared.context import AppContext
from shared.io import load_json
from shared.math import preprocess_ancestor_sets, get_ancestors_inclusive
from shared.paths import OUTPUTS_DIR, ALIAS_TO_CANONICAL_PATH, HPO_LABELS_PATH
from shared.pipeline import PipelineConfig
from shared.result import SimilarityResult

EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
CACHE_BASE_DIR = EVALUATION_DIR / "cache"

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


# ── Data loading ──────────────────────────────────────────────────────────────


def load_test_cases(path: Path) -> list[tuple[list[str], list[str]]]:
    """Load test cases. Returns (hpo_terms, ground_truth) tuples."""
    raw = load_json(path)
    return [(entry[0], entry[1]) for entry in raw]


# ── Patient builder ───────────────────────────────────────────────────────────


def build_patient(
    index: int,
    hpo_terms: list[str],
    ancestor_sets: dict,
) -> PatientProfile:
    """Build PatientProfile with propagated HPO terms."""
    raw_terms = set(hpo_terms)
    propagated = set()
    for term in raw_terms:
        propagated |= get_ancestors_inclusive(term, ancestor_sets)
    return PatientProfile(
        patient_id=f"eval_case_{index:04d}",
        raw_text="",
        hpo_terms=raw_terms,
        propagated_hpo_terms=propagated,
    )


# ── Serialization ─────────────────────────────────────────────────────────────


def serialize_similarity_results(
    results: dict,
) -> dict[str, list[dict]]:
    serialized = {}
    for method, rows in results.items():
        # Handle both list of SimilarityResult and MethodResults objects
        if hasattr(rows, 'rankings'):
            serialized[method] = [r.to_dict() for r in rows.rankings]
        elif isinstance(rows, list):
            serialized[method] = [r.to_dict() for r in rows]
        else:
            serialized[method] = []
    return serialized


# ── Cache helpers ─────────────────────────────────────────────────────────────


def cache_path_for(cache_dir: Path, index: int) -> Path:
    return cache_dir / f"case_{index:04d}.json"


def save_case_cache(
    path: Path,
    index: int,
    hpo_terms: list[str],
    ground_truth: list[str],
    results: dict,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "case_index": index,
        "hpo_terms": sorted(hpo_terms),
        "ground_truth": ground_truth,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "methods_run": sorted(results.keys()),
        "results": results,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Print helpers ─────────────────────────────────────────────────────────────


def print_header(pipeline: str, test_set_path: Path, cache_dir: Path, resume: bool, limit) -> None:
    print(f"\n{'=' * 64}")
    print(f"  RareSim Batch Runner — {pipeline}")
    print(f"{'=' * 64}")
    print(f"  Test set : {test_set_path.name}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Resume   : {resume}")
    print(f"  Limit    : {limit if limit else 'all cases'}")
    print(f"{'=' * 64}\n")


def print_summary(total, processed, skipped, failed, total_time, cache_dir) -> None:
    print(f"\n{'=' * 64}")
    print(f"  Batch complete")
    print(f"{'=' * 64}")
    print(f"  Total    : {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped  : {skipped} (cached)")
    print(f"  Failed   : {failed}")
    if processed > 0:
        print(f"  Time     : {total_time / 60:.1f}min")
        print(f"  Avg/case : {total_time / processed:.1f}s")
    print(f"  Cache    : {cache_dir}")
    print(f"{'=' * 64}\n")


# ── CPU pipeline (semantic/set/tfidf) ─────────────────────────────────────────


def run_cpu_batch(
    test_set_path: Path,
    resume: bool = True,
    config: PipelineConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Run semantic, set-based, and tfidf pipelines on all test cases."""
    from similarity_methods.semantic.pipeline import run as run_semantic
    from similarity_methods.set_based.pipeline import run as run_set_based
    from similarity_methods.tfidf.pipeline import run as run_tfidf

    if config is None:
        config = PipelineConfig()

    test_set_name = test_set_path.stem
    cache_dir = CACHE_BASE_DIR / test_set_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("semantic / set-based / tfidf", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    print("Loading shared context...")
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, config.use_canonical_profiles)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")
    print(f"  HPO labels       : {ctx.app_metadata.n_hpo_labels}")
    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print(f"  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and cache_file.exists():
            skipped += 1
            continue

        patient = build_patient(index, hpo_terms, ancestor_sets)
        print(
            f"[{index + 1:>4}/{total}] case_{index:04d} | "
            f"{len(hpo_terms)} HPO terms ({len(patient.propagated_hpo_terms)} propagated) | "
            f"gt={ground_truth}"
        )

        try:
            start = time.time()
            results = {}
            results.update(run_semantic(patient, SEMANTIC_METHODS, config, ctx))
            results.update(run_set_based(patient, SET_BASED_METHODS, config, ctx))
            results.update(run_tfidf(patient, TFIDF_METHODS, config, ctx))
            elapsed = time.time() - start
            total_time += elapsed

            save_case_cache(
                cache_file, index, hpo_terms, ground_truth,
                serialize_similarity_results(results), elapsed
            )
            processed += 1
            avg = total_time / processed
            remaining = (total - index - 1) * avg
            print(f"           ✓ {elapsed:.1f}s | avg={avg:.1f}s | est. remaining={remaining/60:.1f}min")

        except Exception as e:
            failed += 1
            print(f"           ✗ ERROR: {e}")
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


# ── Transformer pipeline ──────────────────────────────────────────────────────


def run_transformer_batch(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """
    Run transformer pipeline on all test cases.

    Builds embedding cache once then ranks each patient.
    Cache is reused across runs — ranking is near-instant after first build.
    Requires GPU.
    """
    from similarity_methods.transformer.config import MODEL_LIST, CANDIDATE_POOL_SIZE
    from similarity_methods.transformer.retriever import DiseaseRetriever

    test_set_name = test_set_path.stem
    cache_dir = CACHE_BASE_DIR / f"{test_set_name}_transformer"
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
    print("Building/loading transformer embedding cache...")
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

        if resume and cache_file.exists():
            skipped += 1
            continue

        print(f"[{index + 1:>4}/{total}] case_{index:04d} | {len(hpo_terms)} HPO terms | gt={ground_truth}")

        try:
            start = time.time()
            patient_dict = {
                "patient_id": f"eval_case_{index:04d}",
                "raw_text": "",
                "hpo_terms": list(hpo_terms),
            }

            results = {}
            for model_name in MODEL_LIST:
                results[model_name] = retriever.rank(
                    model_name=model_name,
                    patient=patient_dict,
                    top_k=top_k,
                    candidate_pool_size=CANDIDATE_POOL_SIZE,
                )

            elapsed = time.time() - start
            total_time += elapsed
            save_case_cache(cache_file, index, hpo_terms, ground_truth, results, elapsed)
            processed += 1
            avg = total_time / processed
            remaining = (total - index - 1) * avg
            print(f"           ✓ {elapsed:.1f}s | avg={avg:.1f}s | est. remaining={remaining/60:.1f}min")

        except Exception as e:
            failed += 1
            print(f"           ✗ ERROR: {e}")
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


# ── LLM pipeline ──────────────────────────────────────────────────────────────


def run_llm_batch(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """
    Run LLM pipeline on all test cases.

    Each model is loaded, run, then unloaded per case to free GPU memory.
    Warning: slow — ~3 min per case per model. Use --limit for testing.
    Requires GPU.
    """
    from similarity_methods.llm.methods import retrieve_diseases_llm, unload_pipeline
    from similarity_methods.llm.config import LLM_MODEL_LIST

    test_set_name = test_set_path.stem
    cache_dir = CACHE_BASE_DIR / f"{test_set_name}_llm"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("llm", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")
    print(f"Models   : {LLM_MODEL_LIST}")
    est = total * len(LLM_MODEL_LIST) * 3
    print(f"Warning  : LLM is slow. Est. total: {est} min\n")

    hpo_labels = load_json(HPO_LABELS_PATH)
    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and cache_file.exists():
            skipped += 1
            continue

        print(f"[{index + 1:>4}/{total}] case_{index:04d} | {len(hpo_terms)} HPO terms | gt={ground_truth}")

        try:
            start = time.time()
            patient_dict = {
                "patient_id": f"eval_case_{index:04d}",
                "raw_text": "",
                "hpo_terms": list(hpo_terms),
            }

            results = {}
            for model_name in LLM_MODEL_LIST:
                model_results, pipe = retrieve_diseases_llm(
                    patient=patient_dict,
                    hpo_labels=hpo_labels,
                    disease_profiles=ctx.disease_profiles,
                    model_name=model_name,
                    top_k=top_k,
                )
                unload_pipeline(pipe)
                results[model_name] = model_results

            elapsed = time.time() - start
            total_time += elapsed
            save_case_cache(cache_file, index, hpo_terms, ground_truth, results, elapsed)
            processed += 1
            avg = total_time / processed
            remaining = (total - index - 1) * avg
            print(f"           ✓ {elapsed:.1f}s | avg={avg:.1f}s | est. remaining={remaining/60:.1f}min")

        except Exception as e:
            failed += 1
            print(f"           ✗ ERROR: {e}")
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


# ── HPO2Vec pipeline ──────────────────────────────────────────────────────────


def run_hpo2vec_batch(
    test_set_path: Path,
    resume: bool = True,
    limit: int | None = None,
    top_k: int = 10,
) -> Path:
    """
    Run HPO2Vec pipeline on all test cases.

    Loads the pre-trained Word2Vec model from outputs/hpo2vec/hpo2vec_model
    and ranks diseases by cosine similarity of IC-weighted HPO embeddings.
    CPU only — fast after model is loaded.
    """
    from gensim.models import Word2Vec

    # Add pipelines/ to path so hpo2vec_pipeline can be imported
    pipelines_dir = str(PROJECT_ROOT / "pipelines")
    if pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)

    from hpo2vec_pipeline import rank_diseases

    test_set_name = test_set_path.stem
    cache_dir = CACHE_BASE_DIR / f"{test_set_name}_hpo2vec"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_header("hpo2vec", test_set_path, cache_dir, resume, limit)

    cases = load_test_cases(test_set_path)
    if limit:
        cases = cases[:limit]
    total = len(cases)
    print(f"Loaded {total} test cases.\n")

    # Load pre-trained model
    model_path = PROJECT_ROOT / "outputs" / "hpo2vec" / "hpo2vec_model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"HPO2Vec model not found at {model_path}.\n"
            "Train it first by running: python pipelines/hpo2vec_pipeline.py"
        )

    print(f"Loading HPO2Vec model from {model_path}...")
    model = Word2Vec.load(str(model_path))
    print(f"  Vocabulary size: {len(model.wv)} nodes")

    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
    ic_values = load_json(PROJECT_ROOT / "outputs" / "shared" / "information_content.json")

    dummy = PatientProfile("batch_init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)
    print(f"  Disease profiles : {ctx.app_metadata.n_disease_profiles}")

    ancestor_sets = preprocess_ancestor_sets(ctx.ancestors)
    print("  Ready.\n")

    skipped, processed, failed = 0, 0, 0
    total_time = 0.0

    for index, (hpo_terms, ground_truth) in enumerate(cases):
        cache_file = cache_path_for(cache_dir, index)

        if resume and cache_file.exists():
            skipped += 1
            continue

        print(f"[{index + 1:>4}/{total}] case_{index:04d} | {len(hpo_terms)} HPO terms | gt={ground_truth}")

        try:
            start = time.time()

            patient = build_patient(index, hpo_terms, ancestor_sets)
            patient_dict = {
                "patient_id": patient.patient_id,
                "raw_text": "",
                "hpo_terms": list(patient.hpo_terms),
                "propagated_hpo_terms": list(patient.propagated_hpo_terms),
            }

            rankings = rank_diseases(
                disease_profiles=ctx.disease_profiles,
                patient=patient_dict,
                model=model,
                ic_values=ic_values,
                alias_to_canonical=alias_to_canonical,
                use_propagated=True,
                top_k=top_k,
            )

            elapsed = time.time() - start
            total_time += elapsed

            save_case_cache(
                cache_file, index, hpo_terms, ground_truth,
                {"hpo2vec": rankings}, elapsed,
            )
            processed += 1
            avg = total_time / processed
            remaining = (total - index - 1) * avg
            print(f"           ✓ {elapsed:.1f}s | avg={avg:.1f}s | est. remaining={remaining/60:.1f}min")

        except Exception as e:
            failed += 1
            print(f"           ✗ ERROR: {e}")
            (cache_dir / f"case_{index:04d}.error").write_text(f"{type(e).__name__}: {e}")

    print_summary(total, processed, skipped, failed, total_time, cache_dir)
    return cache_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RareSim batch evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to test set JSON file",
    )
    parser.add_argument(
        "--pipeline",
        choices=["cpu", "transformer", "llm", "hpo2vec"],
        default="cpu",
        help="Pipeline to run: cpu (semantic/set/tfidf), transformer, llm, or hpo2vec (default: cpu)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun all cases even if already cached",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N cases (for testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results per method (default: 10)",
    )
    parser.add_argument(
        "--ic-threshold",
        type=float,
        default=1.5,
        help="IC threshold for semantic methods (default: 1.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = not args.no_resume

    if args.pipeline == "cpu":
        config = PipelineConfig(
            top_k=args.top_k,
            ic_threshold=args.ic_threshold,
            use_propagated_terms=True,
            use_canonical_profiles=True,
        )
        run_cpu_batch(
            test_set_path=args.test_set,
            resume=resume,
            config=config,
            limit=args.limit,
        )

    elif args.pipeline == "transformer":
        run_transformer_batch(
            test_set_path=args.test_set,
            resume=resume,
            limit=args.limit,
            top_k=args.top_k,
        )

    elif args.pipeline == "llm":
        run_llm_batch(
            test_set_path=args.test_set,
            resume=resume,
            limit=args.limit,
            top_k=args.top_k,
        )

    elif args.pipeline == "hpo2vec":
        run_hpo2vec_batch(
            test_set_path=args.test_set,
            resume=resume,
            limit=args.limit,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
