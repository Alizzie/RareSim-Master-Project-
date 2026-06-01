"""Run Dx29 Phrank on benchmark datasets and evaluate results."""

import requests
import argparse
import time
from pathlib import Path
from utils import (
    resolve_datasets,
    load_all_datasets,
    save_summary_tsv,
    compute_stats,
    print_stats,
)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "PhenoBrainBenchmarkDatasets"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Dx29 using the Phrank algorithm on benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing dataset JSON files (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Datasets to run (default: all)",
    )
    p.add_argument(
        "--host",
        default="http://localhost:8080",
        help="Dx29 API host (default: http://localhost:8080)",
    )
    p.add_argument(
        "--lang",
        default="en",
        help="Language for Dx29 API (default: en)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=1000,
        help="Number of top predictions to retrieve from Dx29 (default: 1000)",
    )
    return p.parse_args()


def predict_case(
    hpo_list: list, topk: int, host: str, lang: str
) -> tuple[bool, list | None]:
    """Predict diseases for a case with given HPO terms using Dx29 API."""
    url = f"{host}/api/v1/Diagnosis/phrank"
    params = {"skip": 0, "count": topk, "lang": lang, "source": "all"}
    json_body = {"symptoms": hpo_list, "genes": []}

    response = requests.post(url, params=params, json=json_body, timeout=30)

    if response.status_code == 200:
        data = response.json()
        return True, list(data)
    else:
        return False, None


def get_disease_ranking(results: list[dict]) -> dict:
    """Extract disease ranking from prediction results.

    Returns a dict mapping rank index -> (disease_id, score).
    """
    return {
        i: (item.get("id"), item.get("scoreDx29", 0.0))
        for i, item in enumerate(results)
    }


def summarize_ranking(
    case_id: str,
    ranking: dict,
    ground_truth: list,
    n_hpo: int,
    status: bool,
    query_time_sec: float | None,
    rows: list,
):
    """Summarize the ranking results for a case and append a row to the accumulator."""
    rank_found, matched_id, sim_score, disease_codes = None, None, None, None

    for rank, (disease_id, score) in ranking.items():
        if disease_id is None:
            continue
        for ground_id in ground_truth:
            if ground_id == disease_id:
                rank_found, matched_id, disease_codes, sim_score = (
                    rank + 1,
                    ground_id,
                    disease_id,
                    score,
                )
                break
        if rank_found is not None:
            break

    if rank_found is None:
        print(f"  {case_id}: ground truth '{ground_truth}' not found in predictions.")

    rows.append(
        {
            "case_id": case_id,
            "n_hpo": n_hpo,
            "confirmed_diseases": disease_codes if disease_codes else "None",
            "rank": rank_found if rank_found is not None else "None",
            "matched_id": matched_id if matched_id is not None else "None",
            "score": f"{sim_score:.4f}" if sim_score is not None else "None",
            "status": bool(status),
            "query_time_sec": (
                f"{query_time_sec:.3f}" if query_time_sec is not None else "None"
            ),
        }
    )

    print(
        f"  {case_id}: rank={rank_found}, matched_id={matched_id}, "
        f"n_hpo={n_hpo}, score={sim_score}, "
        f"status={status}, query_time_sec={query_time_sec:.3f}"
    )

    return rank_found, matched_id, sim_score


def run_dataset(name: str, cases: list, args) -> list[dict]:
    """Run the full Dx29 pipeline for one dataset. Returns summary list."""
    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    rows = []
    for case_id, hpo_ids, ground_truth in entries:
        start_time = time.time()
        status, results = predict_case(hpo_ids, args.topk, args.host, args.lang)
        query_time_sec = time.time() - start_time

        disease_ranking = get_disease_ranking(results) if status and results else {}
        summarize_ranking(
            case_id,
            disease_ranking,
            ground_truth,
            len(hpo_ids),
            status,
            query_time_sec,
            rows,
        )

    return rows


def main():
    args = parse_args()
    workdir = SCRIPT_DIR / "dx29_phrank_benchmarks"
    workdir.mkdir(parents=True, exist_ok=True)

    selected = resolve_datasets(args.data_dir, args.datasets)
    if not selected:
        print("No datasets to process. Exiting.")
        return

    all_cases = load_all_datasets(args.data_dir, selected)

    all_summaries = {}
    for dataset_name, cases in all_cases.items():
        print(f"Processing dataset: {dataset_name} with {len(cases)} cases")
        summary = run_dataset(dataset_name, cases, args)
        all_summaries[dataset_name] = summary
        save_summary_tsv(summary, workdir / f"{dataset_name}_summary.tsv")

    print("Generating final summary statistics")
    for dataset_name, summary in all_summaries.items():
        out_path = workdir / f"{dataset_name}_stats.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            print_stats(dataset_name, compute_stats(summary), f)
        print(f"  Stats written to {out_path}")


if __name__ == "__main__":
    main()
