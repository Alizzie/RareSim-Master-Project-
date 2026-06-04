"""
Text vs HPO Input Experiment
=============================
Compares two input modes for transformer-based disease retrieval:
  1. HPO terms  → converted to phenotype labels → embedded → ranked
  2. Raw text   → embedded directly              → ranked

Models tested:
  - PubMedBERT  (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
  - ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
  - MiniLM      (sentence-transformers/all-MiniLM-L6-v2)
  - SapBERT     (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
  - BioBERT     (dmis-lab/biobert-v1.1)

Test set: 100 medical cases with ground truth ORPHA codes

Usage:
    # Run HPO input mode - check available gpu then choose one for CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=6 python experiments/text_vs_hpo/scripts/run_experiment.py --mode hpo

    # Run raw text input mode
    CUDA_VISIBLE_DEVICES=6 python experiments/text_vs_hpo/scripts/run_experiment.py --mode text

    # Run both
    CUDA_VISIBLE_DEVICES=6 python experiments/text_vs_hpo/scripts/run_experiment.py --mode both
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Project path setup ────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.schemas import PatientProfile
from shared.context import AppContext
from shared.io import load_json
from shared.paths import ALIAS_TO_CANONICAL_PATH, HPO_LABELS_PATH
from similarity_methods.transformer.config import CANDIDATE_POOL_SIZE, MODEL_LIST
from similarity_methods.transformer.retriever import DiseaseRetriever

RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K = 10


# ── Data loading ──────────────────────────────────────────────────────────────


def load_hpo_cases(path: Path) -> list[dict]:
    """
    Load HPO test cases.
    Format: [[hpo_terms], [disease_codes]]
    Returns list of {patient_id, hpo_terms, ground_truth}
    """
    raw = load_json(path)
    cases = []
    for i, entry in enumerate(raw):
        hpo_terms, ground_truth = entry[0], entry[1]
        cases.append({
            "patient_id": f"hpo_case_{i:04d}",
            "raw_text": "",
            "hpo_terms": hpo_terms,
            "ground_truth": ground_truth,
        })
    return cases


def load_raw_text_cases(path: Path) -> list[dict]:
    """
    Load raw text test cases.
    Format: {orpha_code: clinical_text}
    Returns list of {patient_id, raw_text, hpo_terms, ground_truth}
    """
    raw = load_json(path)
    cases = []
    for i, (orpha_code, text) in enumerate(raw.items()):
        cases.append({
            "patient_id": f"text_case_{i:04d}",
            "raw_text": text.strip(),
            "hpo_terms": [],  # no HPO terms — raw text only
            "ground_truth": [f"ORPHA:{orpha_code}"],
        })
    return cases


# ── Metrics ───────────────────────────────────────────────────────────────────


def find_rank(ground_truth: list[str], results: list[dict]) -> int | None:
    """Find the best rank of any ground truth disease in results."""
    gt_set = set(ground_truth)
    for r in results:
        if r.get("canonical_disease_id") in gt_set:
            return r["rank"]
    return None


def compute_metrics(ranks: list[int | None], top_k: int = TOP_K) -> dict:
    """Compute Recall@1/5/10, MRR, NDCG@10."""
    import math
    n = len(ranks)
    if n == 0:
        return {}
    return {
        "recall_1":  round(sum(1 for r in ranks if r == 1) / n, 4),
        "recall_5":  round(sum(1 for r in ranks if r and r <= 5) / n, 4),
        "recall_10": round(sum(1 for r in ranks if r and r <= 10) / n, 4),
        "mrr":       round(sum(1/r for r in ranks if r) / n, 4),
        "ndcg":      round(sum(1/math.log2(r+1) for r in ranks if r and r <= top_k) / n, 4),
        "found":     sum(1 for r in ranks if r),
        "total":     n,
    }


# ── Runner ────────────────────────────────────────────────────────────────────


def run_mode(
    mode: str,
    cases: list[dict],
    retriever: DiseaseRetriever,
) -> dict:
    """
    Run all models on given cases and collect results.
    Returns {model_name: {metrics, per_case_ranks}}
    """
    print(f"\n{'='*60}")
    print(f"  Running mode: {mode.upper()} ({len(cases)} cases)")
    print(f"{'='*60}")

    all_results = {}

    for model_name in MODEL_LIST:
        print(f"\n  Model: {model_name}")
        ranks = []
        per_case = []
        start = time.time()

        for case in cases:
            try:
                results = retriever.rank(
                    model_name=model_name,
                    patient=case,
                    top_k=TOP_K,
                    candidate_pool_size=CANDIDATE_POOL_SIZE,
                )
                rank = find_rank(case["ground_truth"], results)
                ranks.append(rank)
                per_case.append({
                    "patient_id": case["patient_id"],
                    "ground_truth": case["ground_truth"],
                    "rank": rank,
                    "top_results": [
                        {"rank": r["rank"], "disease_id": r["canonical_disease_id"], "label": r["label"]}
                        for r in results
                    ],
                })
            except Exception as e:
                print(f"    ERROR on {case['patient_id']}: {e}")
                ranks.append(None)
                per_case.append({
                    "patient_id": case["patient_id"],
                    "ground_truth": case["ground_truth"],
                    "rank": None,
                    "error": str(e),
                })

        elapsed = time.time() - start
        metrics = compute_metrics(ranks)
        metrics["elapsed_seconds"] = round(elapsed, 1)

        all_results[model_name] = {
            "metrics": metrics,
            "per_case": per_case,
        }

        print(f"    R@1={metrics['recall_1']:.4f} | R@5={metrics['recall_5']:.4f} | "
              f"R@10={metrics['recall_10']:.4f} | MRR={metrics['mrr']:.4f} | "
              f"Found={metrics['found']}/{metrics['total']} | {elapsed:.1f}s")

    return all_results


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Text vs HPO input experiment")
    parser.add_argument("--mode", choices=["hpo", "text", "both"], default="both")
    args = parser.parse_args()

    # Load shared resources
    print("Loading resources...")
    hpo_labels = load_json(HPO_LABELS_PATH)
    alias_to_canonical = load_json(ALIAS_TO_CANONICAL_PATH)
    dummy = PatientProfile("init", "", set(), set())
    ctx = AppContext.load(dummy, use_canonical_profiles=True)

    # Build retriever (shared across both modes)
    retriever = DiseaseRetriever(
        disease_profiles=ctx.disease_profiles,
        hpo_labels=hpo_labels,
        alias_to_canonical=alias_to_canonical,
        model_list=MODEL_LIST,
    )
    print("Warming up models (loading embedding caches)...")
    retriever.warmup(preload_models=True)

    experiment_results = {}

    # HPO mode
    if args.mode in ("hpo", "both"):
        hpo_cases = load_hpo_cases(EXPERIMENT_DIR / "test_data" / "hpo_cases.json")
        experiment_results["hpo"] = run_mode("hpo", hpo_cases, retriever)

    # Text mode
    if args.mode in ("text", "both"):
        text_cases = load_raw_text_cases(EXPERIMENT_DIR / "test_data" / "raw_text_cases.json")
        experiment_results["text"] = run_mode("text", text_cases, retriever)

    # Save full results
    out_path = RESULTS_DIR / "experiment_results.json"
    with out_path.open("w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nSaved results → {out_path}")

    # Print comparison summary
    print_summary(experiment_results)


def print_summary(results: dict):
    print(f"\n{'='*80}")
    print(f"  Text vs HPO Input — Comparison Summary ({TOP_K} cases, top-{TOP_K})")
    print(f"{'='*80}")
    print(f"  {'Model':<50} {'Mode':<6} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}  Found")
    print(f"  {'-'*78}")

    for model_name in MODEL_LIST:
        short = model_name.split("/")[-1][:48]
        for mode in ["hpo", "text"]:
            if mode not in results:
                continue
            m = results[mode][model_name]["metrics"]
            print(f"  {short:<50} {mode:<6} "
                  f"{m['recall_1']:>6.4f} {m['recall_5']:>6.4f} "
                  f"{m['recall_10']:>6.4f} {m['mrr']:>6.4f}  "
                  f"{m['found']}/{m['total']}")
        print()

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
