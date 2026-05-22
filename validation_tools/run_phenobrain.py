"""Run Phenobrain on benchmark datasets and evaluate results."""

import requests
import time
import argparse
import csv
from pathlib import Path
from utils import load_all_datasets, DATASET_NAMES
import os

TOP_K = 10


def parse_args():
    p = argparse.ArgumentParser(description="Run Phenobrain on benchmark datasets.")
    p.add_argument(
        "--data-dir", required=True, help="Directory containing the 6 JSON files"
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="Datasets to run (default: all)",
    )

    return p.parse_args()


def predict_case(hpo_list: list):
    """Predict diseases for a case with given HPO terms using Phenobrain API."""
    url = "https://www.phenobrain.cs.tsinghua.edu.cn/predict"
    params = {"model": "Ensemble", "hpoList[]": hpo_list, "topk": TOP_K}

    response = requests.get(url, params=params, timeout=30)
    task_id = response.json().get("TASK_ID", None)
    status = response.status_code == 200 and task_id is not None
    return status, task_id


def retrieve_query_results(task_id: str):
    """Retrieve prediction results for a given task ID."""
    url = f"https://www.phenobrain.cs.tsinghua.edu.cn/query-predict-result?taskId={task_id}"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def wait_for_results(task_id: str, poll_interval: float = 3.0, timeout: float = 300.0):
    """Wait for prediction results to be ready by polling the API."""
    start_time = time.time()
    while True:
        data = retrieve_query_results(task_id)
        if data is None:
            return RuntimeError(f"Failed to retrieve results for task ID {task_id}.")

        state = data.get("state")
        print(f"State: {state}")

        if state == "SUCCESS":
            return data.get("result")

        if state not in ("MODEL_INIT", "MODEL_PREDICT"):
            raise RuntimeError(f"Unexpected state '{state}' for task ID {task_id}.")

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Results for task ID {task_id} not available after {timeout} seconds."
            )

        time.sleep(poll_interval)


def create_RD_code_mapper(rd_codes: list):
    """Create a mapping from RD codes to OMIM IDs. If not exist, ORPHA"""

    url = "https://www.phenobrain.cs.tsinghua.edu.cn/disease-list-detail"
    payload = {"diseaseList": rd_codes}

    response = requests.post(url, json=payload, timeout=30)

    if response.status_code == 200:
        rd_to_codes = {}
        output = dict(response.json())

        for rd_code, details in output.items():
            sources_codes = details.get("SOURCE_CODES", [])
            disease_codes = ";".join(sources_codes)
            rd_to_codes[rd_code] = disease_codes
        return rd_to_codes

    else:
        raise RuntimeError(
            f"Failed to retrieve disease details for RD codes: {rd_codes}"
        )


def get_disease_ranking(results: list[dict], code_mapper: dict):
    """Extract disease list details from prediction results."""

    ranking = {}
    for i, item in enumerate(results):
        disease_code = code_mapper.get(item.get("CODE", None), None)
        score = item.get("SCORE", 0.0)

        ranking[i] = (disease_code, score)

    return ranking


def summarize_ranking(
    case_id: str,
    ranking: dict,
    ground_truth: list,
    n_hpo: int,
    status: bool,
    rows: list,  # accumulator passed in from main
):
    """Summarize the ranking results for a case and append a row to the accumulator."""

    rank_found, matched_id, sim_score = None, None, None
    disease_codes = None

    for rank, (disease_code, score) in ranking.items():
        for ground_id in ground_truth:
            if ground_id in disease_code.split(";"):
                rank_found, matched_id, disease_codes, sim_score = (
                    rank + 1,
                    ground_id,
                    disease_code,
                    score,
                )
                break

    if rank_found is None:
        print(f"Ground truth '{ground_truth}' not found in top predictions.")

    rows.append(
        {
            "case_id": case_id,
            "n_hpo": n_hpo,
            "confirmed_diseases": disease_codes if disease_codes else "NA",
            "rank": rank_found if rank_found is not None else "NA",
            "matched_id": matched_id if matched_id is not None else "NA",
            "score": f"{sim_score:.4f}" if sim_score is not None else "NA",
            "status": bool(status),
        }
    )

    return rank_found, matched_id, sim_score


def write_results(rows: list, output_file: str):
    """Write all accumulated rows to a TSV file, overwriting any existing file."""
    headers = [
        "case_id",
        "n_hpo",
        "confirmed_diseases",
        "rank",
        "matched_id",
        "score",
        "status",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:  # "w" overwrites
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()
    all_cases = load_all_datasets(Path(args.data_dir), args.datasets)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "phenobrain_benchmarks")
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name, cases in all_cases.items():
        print(f"Processing dataset: {dataset_name} with {len(cases)} cases")
        rows = []

        for i, (hpo_ids, ground_truth) in enumerate(cases):
            status, task_id = predict_case(hpo_ids)
            print(f"Task ID: {task_id}, Status: {status}")

            if status:
                results = wait_for_results(task_id)
                rd_codes = [item.get("CODE") for item in results]
                rd_code_mapping = create_RD_code_mapper(rd_codes)
                disease_ranking = get_disease_ranking(results, rd_code_mapping)

                case_id = f"{dataset_name}_case_{i:04d}"
                n_hpo = len(hpo_ids)

                summarize_ranking(
                    case_id, disease_ranking, ground_truth, n_hpo, status, rows
                )
            else:
                summarize_ranking(case_id, {}, ground_truth, len(hpo_ids), status, rows)

        write_results(rows, output_file=f"{results_dir}/{dataset_name}_results.tsv")
