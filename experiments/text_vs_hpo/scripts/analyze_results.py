"""
Analyze experiment results.

Usage:
    python experiments/text_vs_hpo/scripts/analyze_results.py

Outputs:
    results/summary.txt
"""

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"


def analyze():
    results_path = RESULTS_DIR / "experiment_results.json"
    if not results_path.exists():
        print("No results found. Run run_experiment.py first.")
        return

    with results_path.open() as f:
        results = json.load(f)

    modes  = list(results.keys())
    models = list(results[modes[0]].keys())

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  Detailed Analysis — Text vs HPO Input")
    lines.append(f"{'='*80}")

    for model_name in models:
        short = model_name.split("/")[-1]
        lines.append(f"\n  {short}")
        lines.append(f"  {'-'*60}")
        for mode in modes:
            m = results[mode][model_name]["metrics"]
            lines.append(
                f"  [{mode.upper():4}] R@1={m['recall_1']:.4f} | R@5={m['recall_5']:.4f} | "
                f"R@10={m['recall_10']:.4f} | MRR={m['mrr']:.4f} | Found={m['found']}/{m['total']}"
            )
        if "hpo" in results and "text" in results:
            hpo_m = results["hpo"][model_name]["metrics"]
            txt_m = results["text"][model_name]["metrics"]
            delta_r10 = hpo_m["recall_10"] - txt_m["recall_10"]
            delta_mrr = hpo_m["mrr"] - txt_m["mrr"]
            lines.append(f"  [Δ HPO-text] R@10={delta_r10:+.4f} | MRR={delta_mrr:+.4f}")

    lines.append(f"\n{'='*80}")
    summary = "\n".join(lines)
    print(summary)

    txt_path = RESULTS_DIR / "summary.txt"
    txt_path.write_text(summary)
    print(f"\nSaved → {txt_path}")


if __name__ == "__main__":
    analyze()
    