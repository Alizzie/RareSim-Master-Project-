"""Save full PhenoBrain API rankings for the external-RRF experiment.

This script does NOT modify the normal PhenoBrain benchmark outputs.
Full per-case rankings are written to:

    outputs/ensemble_external_experiment/phenobrain/<DATASET>/

Existing files are skipped, so the script can safely be resumed.
"""

import argparse
import json
import time
from pathlib import Path

from _utils import resolve_datasets, load_all_datasets
from run_phenobrain import (
    predict_case,
    wait_for_results,
    create_RD_code_mapper,
)
from raresim.utils.paths import OUTPUTS_DIR, DATASET_DIR




DEFAULT_DATA_DIR = DATASET_DIR / "phenobrain_testdata"

EXPERIMENT_DIR = (
    OUTPUTS_DIR / "ensemble_external_experiment" / "phenobrain"
)

DEFAULT_DATASETS = [
    "PUMCH_L",
    "MME",
    "PUMCH-ADM",
    "RAMEDIS",
    "HMS",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=200,
    )

    return parser.parse_args()


def save_case(
    dataset: str,
    case_index: int,
    hpo_ids: list[str],
    ground_truth: list[str],
    ranking: list[dict],
    query_time: float,
):
    dataset_dir = EXPERIMENT_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    path = dataset_dir / f"case_{case_index:04d}.json"

    payload = {
        "case_index": case_index,
        "hpo_terms": hpo_ids,
        "ground_truth": ground_truth,
        "query_time_sec": query_time,
        "results": ranking,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_dataset(dataset: str, cases: list, topk: int):
    dataset_dir = EXPERIMENT_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset}: {len(cases)} cases ===")

    for case_index, (hpo_ids, ground_truth) in enumerate(cases):
        output_path = dataset_dir / f"case_{case_index:04d}.json"

        if output_path.exists():
            print(f"  case_{case_index:04d}: cached")
            continue

        start = time.time()

        status, task_id = predict_case(hpo_ids, topk)

        if not status or task_id is None:
            print(f"  case_{case_index:04d}: submission failed")
            continue

        results = wait_for_results(task_id)
        query_time = time.time() - start

        if not results:
            print(f"  case_{case_index:04d}: empty ranking")
            continue

        rd_codes = [
            item.get("CODE")
            for item in results
            if item.get("CODE")
        ]

        mapper = create_RD_code_mapper(rd_codes)

        ranking = []

        for rank, item in enumerate(results, start=1):
            rd_code = item.get("CODE")

            if not rd_code:
                continue

            mapped = mapper.get(rd_code, "")

            # PhenoBrain may map one RD disease to several source IDs,
            # e.g. CCRD:33;OMIM:263800;ORPHA:358.
            disease_ids = [
                value.strip()
                for value in mapped.split(";")
                if value.strip()
            ]

            ranking.append(
                {
                    "rank": rank,
                    "rd_code": rd_code,
                    "disease_ids": disease_ids,
                    "score": item.get("SCORE"),
                }
            )

        save_case(
            dataset,
            case_index,
            hpo_ids,
            ground_truth,
            ranking,
            query_time,
        )

        print(
            f"  case_{case_index:04d}: "
            f"{len(ranking)} candidates saved"
        )


def main():
    args = parse_args()

    selected = resolve_datasets(
        args.data_dir,
        args.datasets,
    )

    all_cases = load_all_datasets(
        args.data_dir,
        selected,
    )

    for dataset, cases in all_cases.items():
        run_dataset(dataset, cases, args.topk)


if __name__ == "__main__":
    main()
