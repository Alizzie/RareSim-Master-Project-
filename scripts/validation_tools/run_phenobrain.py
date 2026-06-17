"""Run Phenobrain on benchmark datasets and evaluate results."""

import requests
import time
import argparse
from pathlib import Path
from _utils import (
    resolve_datasets,
    load_all_datasets,
    save_summary_tsv,
    compute_stats,
    print_stats,
)
from raresim.utils.paths import OUTPUTS_DIR, DATASET_DIR

VAL_OUTPUTS_DIR = OUTPUTS_DIR / "validation_tools"
DEFAULT_DATA_DIR = DATASET_DIR / "PhenoBrainBenchmarkDatasets"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Phenobrain on benchmark datasets.",
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
        "--topk",
        type=int,
        default=200,
        help="Number of top predictions to retrieve from Phenobrain (default & max: 200)",
    )
    return p.parse_args()

def predict_case(hpo_list: list, topk: int, max_retries : int =3, retry_delay : int =5)  -> tuple[bool, str | None]:
    """Predict diseases for a case with given HPO terms using Phenobrain API. Max 3 retries on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            url = "https://www.phenobrain.cs.tsinghua.edu.cn/predict"
            params = {"model": "Ensemble", "hpoList[]": hpo_list, "topk": topk}

            response = requests.get(url, params=params, timeout=30)

            if not response.ok:
                print(f"[Attempt {attempt}/{max_retries}] Request failed: {response.status_code} {response.text}")
            elif not response.text.strip():
                print(f"[Attempt {attempt}/{max_retries}] Empty response body (status {response.status_code})")
            else:
                task_id = response.json().get("TASK_ID", None)
                return True, task_id

        except (requests.exceptions.JSONDecodeError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"[Attempt {attempt}/{max_retries}] Exception: {e}")

        if attempt < max_retries:
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    print(f"All {max_retries} attempts failed for this case.")
    return False, None


def retrieve_query_results(task_id: str, max_retries: int = 3, retry_delay: int = 5):
    """Retrieve prediction results for a given task ID."""
    url = f"https://www.phenobrain.cs.tsinghua.edu.cn/query-predict-result?taskId={task_id}"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                if not response.text.strip():
                    print(f"[Attempt {attempt}/{max_retries}] Empty response body")
                else:
                    return response.json()
            else:
                print(f"[Attempt {attempt}/{max_retries}] Bad status: {response.status_code}")

        except (requests.exceptions.JSONDecodeError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"[Attempt {attempt}/{max_retries}] Exception: {e}")

        if attempt < max_retries:
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    return None 


def wait_for_results(task_id: str, poll_interval: float = 3.0, timeout: float = 300.0):
    """Wait for prediction results to be ready by polling the API."""
    start_time = time.time()
    while True:
        data = retrieve_query_results(task_id)
        if data is None:
            raise RuntimeError(f"Failed to retrieve results for task ID {task_id}.")

        state = data.get("state")
        print(f"  State: {state}")

        if state == "SUCCESS":
            return data.get("result")

        if state not in ("MODEL_INIT", "MODEL_PREDICT", "PENDING"):
            raise RuntimeError(f"Unexpected state '{state}' for task ID {task_id}.")

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Results for task ID {task_id} not available after {timeout} seconds."
            )

        time.sleep(poll_interval)


def create_RD_code_mapper(rd_codes: list, max_retries: int = 3, retry_delay: int = 5) -> dict:
    """Create a mapping from RD codes to OMIM IDs. If not exist, ORPHA."""
    url = "https://www.phenobrain.cs.tsinghua.edu.cn/disease-list-detail"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json={"diseaseList": rd_codes}, timeout=30)

            if response.status_code == 200:
                if not response.text.strip():
                    print(f"[Attempt {attempt}/{max_retries}] Empty response body")
                else:
                    return {
                        rd_code: ";".join(details.get("SOURCE_CODES", []))
                        for rd_code, details in response.json().items()
                    }
            else:
                print(f"[Attempt {attempt}/{max_retries}] Bad status: {response.status_code} {response.text}")

        except (requests.exceptions.JSONDecodeError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"[Attempt {attempt}/{max_retries}] Exception: {e}")

        if attempt < max_retries:
            print(f"Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise RuntimeError(f"Failed to retrieve disease details for RD codes after {max_retries} attempts: {rd_codes}")


def get_disease_ranking(results: list[dict], code_mapper: dict) -> dict:
    """Extract disease ranking from prediction results.

    Returns a dict mapping rank index -> (disease_codes_str, score).
    """
    return {
        i: (code_mapper.get(item.get("CODE"), None), item.get("SCORE", 0.0))
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

    for rank, (disease_code, score) in ranking.items():
        if disease_code is None:
            continue
        for ground_id in ground_truth:
            if ground_id in disease_code.split(";"):
                rank_found, matched_id, disease_codes, sim_score = (
                    rank + 1,
                    ground_id,
                    disease_code,
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
    """Run the full Phenobrain pipeline for one dataset. Returns summary list."""
    entries = [
        (f"{name}_case_{i:04d}", hpo_ids, diseases)
        for i, (hpo_ids, diseases) in enumerate(cases)
    ]

    rows = []
    for case_id, hpo_ids, ground_truth in entries:
        start_time = time.time()
        status, task_id = predict_case(hpo_ids, args.topk)
        print(f"  {case_id}: task_id={task_id}, status={status}")

        if status and task_id is not None:
            results = wait_for_results(task_id)
            query_time_sec = time.time() - start_time
            rd_codes = [item.get("CODE") for item in results]
            rd_code_mapping = create_RD_code_mapper(rd_codes)
            disease_ranking = get_disease_ranking(results, rd_code_mapping)
        else:
            query_time_sec = time.time() - start_time
            disease_ranking = {}

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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    workdir = VAL_OUTPUTS_DIR / "phenobrain_benchmarks"
    workdir.mkdir(parents=True, exist_ok=True)

    selected = resolve_datasets(args.data_dir, args.datasets)
    if not selected:
        print("No datasets selected, exiting.")
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
