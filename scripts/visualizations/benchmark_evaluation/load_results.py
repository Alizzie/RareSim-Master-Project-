"""
Loading and metric computation for RareSim and validation-tool evaluations.

Supported RareSim inputs:
    evaluation/HMS/HMS_summary.tsv
    evaluation/HMS/HMS_evaluation.json

Supported validation-tool inputs:
    validation_tools/lirical_benchmarks/hms_summary.tsv
    validation_tools/phenobrain_benchmarks/mme_summary.tsv
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts.visualizations.benchmark_evaluation.config import (
    DATASET_NAME_MAP,
    METHOD_LABELS,
    VALIDATION_TOOL_LABELS,
)


def normalize_dataset_name(value: str) -> str:
    """Normalize dataset names from folder names and filenames."""
    cleaned = value.strip()
    cleaned = re.sub(r"\.tsv$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\.json$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_evaluation$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_summary.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"_stats$", "", cleaned, flags=re.IGNORECASE)

    key = cleaned.lower()
    return DATASET_NAME_MAP.get(key, cleaned.upper())


def clean_method_label(method: str) -> str:
    """Return a shorter readable method name."""
    return METHOD_LABELS.get(method, method)


def validation_tool_label(path: Path) -> str:
    """
    Extract validation tool name from parent folder.

    Example:
        validation_tools/lirical_benchmarks/hms_summary.tsv -> LIRICAL
    """
    folder = path.parent.name
    folder = re.sub(r"_benchmarks$", "", folder, flags=re.IGNORECASE)
    folder = re.sub(r"_benchmark$", "", folder, flags=re.IGNORECASE)
    return VALIDATION_TOOL_LABELS.get(folder.lower(), folder.replace("_", " ").title())


def method_family(method: str) -> str:  # pylint: disable=too-many-return-statements
    """Assign RareSim method to a broad method family."""
    method_lower = method.lower()

    if method.startswith("ensemble"):
        return "Ensemble"
    if method.startswith("semantic_"):
        return "Semantic"
    if method.startswith("set_"):
        return "Set-based"
    if method in {"tfidf", "tfidf_cosine"}:
        return "TF-IDF"
    if method == "hpo2vec":
        return "HPO2Vec"
    if "mistral" in method_lower or "llm" in method_lower:
        return "LLM"
    if "autoencoder" in method_lower:
        return "Autoencoder"
    if any(term in method_lower for term in ["bert", "minilm", "transformer"]):
        return "Transformer encoder"

    return "Other RareSim method"


def system_type_for_raresim(method: str) -> str:
    """Separate ensembles from individual RareSim methods."""
    if method.startswith("ensemble"):
        return "Ensemble"
    return "RareSim method"


def read_tsv(path: Path) -> pd.DataFrame:
    """Read TSV file."""
    return pd.read_csv(path, sep="\t")


def compute_metrics_from_rank_rows(  # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    df: pd.DataFrame,
    dataset: str,
    method_col: str,
    method_override: str | None,
    system_type: str,
    source_file: Path,
    source_priority: int,
) -> pd.DataFrame:
    """
    Compute method-level metrics from rows containing ranks.

    Required:
        rank

    Recommended:
        case_id, method, query_time_sec
    """
    if "rank" not in df.columns:
        raise ValueError(f"{source_file} has no 'rank' column.")

    df = df.copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    if "case_id" not in df.columns:
        df["case_id"] = [f"case_{idx:04d}" for idx in range(len(df))]

    if "query_time_sec" not in df.columns:
        df["query_time_sec"] = np.nan
    else:
        df["query_time_sec"] = pd.to_numeric(df["query_time_sec"], errors="coerce")

    if method_override is not None:
        df["_method_name"] = method_override
    elif method_col in df.columns:
        df["_method_name"] = df[method_col].astype(str)
    else:
        df["_method_name"] = source_file.stem

    n_cases = df["case_id"].nunique()
    rows = []

    for method, group in df.groupby("_method_name", sort=False):
        method_name = str(method)

        ranks = pd.to_numeric(group["rank"], errors="coerce").to_numpy(dtype=float)
        found = ~np.isnan(ranks)

        reciprocal_rank = np.where(found, 1.0 / ranks, 0.0)
        ndcg_10 = np.where(
            found & (ranks <= 10),
            1.0 / np.log2(ranks + 1.0),
            0.0,
        )

        if system_type == "Validation tool":
            display_name = method_name
            family = "Validation tool"
            final_system_type = "Validation tool"
        else:
            display_name = clean_method_label(method_name)
            family = method_family(method_name)
            final_system_type = system_type_for_raresim(method_name)

        rows.append(
            {
                "dataset": dataset,
                "method": method_name,
                "method_label": display_name,
                "method_family": family,
                "system_type": final_system_type,
                "n_cases": n_cases,
                "found_count": int(found.sum()),
                "found_rate": float(found.mean()),
                "R@1": float((ranks <= 1).mean()),
                "R@3": float((ranks <= 3).mean()),
                "R@5": float((ranks <= 5).mean()),
                "R@10": float((ranks <= 10).mean()),
                "MRR": float(reciprocal_rank.mean()),
                "NDCG@10": float(ndcg_10.mean()),
                "avg_query_time_sec": (
                    float(group["query_time_sec"].mean())
                    if group["query_time_sec"].notna().any()
                    else np.nan
                ),
                "median_rank_found": (
                    float(np.nanmedian(ranks)) if found.any() else np.nan
                ),
                "source_file": str(source_file),
                "source_priority": source_priority,
            }
        )

    return pd.DataFrame(rows)


def load_raresim_tsv(path: Path) -> pd.DataFrame:
    """Load one RareSim summary TSV file."""
    dataset = normalize_dataset_name(path.parent.name)
    df = read_tsv(path)

    return compute_metrics_from_rank_rows(
        df=df,
        dataset=dataset,
        method_col="method",
        method_override=None,
        system_type="RareSim method",
        source_file=path,
        source_priority=2,
    )


def load_validation_tsv(path: Path) -> pd.DataFrame:
    """
    Load one validation-tool TSV file.

    For validation folders:
        parent folder = validation tool
        filename = dataset
    """
    dataset = normalize_dataset_name(path.name)
    tool = validation_tool_label(path)
    df = read_tsv(path)

    return compute_metrics_from_rank_rows(
        df=df,
        dataset=dataset,
        method_col="method",
        method_override=tool,
        system_type="Validation tool",
        source_file=path,
        source_priority=2,
    )


def load_raresim_json(path: Path) -> pd.DataFrame:
    """
    Load one RareSim evaluation JSON file.

    Expected keys:
        n_cases
        method_metrics
        method_avg_seconds
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    dataset = normalize_dataset_name(path.parent.name)
    if dataset in {"EVALUATION", "RESULTS", "OUTPUTS"}:
        dataset = normalize_dataset_name(path.name)

    n_cases = data.get("n_cases")
    method_metrics = data.get("method_metrics", {})
    method_avg_seconds = data.get("method_avg_seconds", {})

    rows = []

    for method, metrics in method_metrics.items():
        found = metrics.get("found", np.nan)
        avg_seconds = method_avg_seconds.get(method, np.nan)

        if pd.isna(avg_seconds) and method == "tfidf_cosine":
            avg_seconds = method_avg_seconds.get("tfidf", np.nan)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_label": clean_method_label(method),
                "method_family": method_family(method),
                "system_type": system_type_for_raresim(method),
                "n_cases": n_cases,
                "found_count": found,
                "found_rate": found / n_cases if n_cases else np.nan,
                "R@1": metrics.get("recall_1", np.nan),
                "R@3": metrics.get("recall_3", np.nan),
                "R@5": metrics.get("recall_5", np.nan),
                "R@10": metrics.get("recall_10", np.nan),
                "R@20": metrics.get("recall_20", np.nan),
                "MRR": metrics.get("mrr", np.nan),
                "NDCG@10": metrics.get("ndcg", np.nan),
                "avg_query_time_sec": avg_seconds,
                "median_rank_found": metrics.get("median_rank", np.nan),
                "source_file": str(path),
                "source_priority": 1,
            }
        )

    return pd.DataFrame(rows)


def load_raresim_results(raresim_dir: Path | None) -> pd.DataFrame:
    """Load RareSim JSON and TSV results."""
    if raresim_dir is None or not raresim_dir.exists():
        return pd.DataFrame()

    frames = []

    for json_path in sorted(raresim_dir.rglob("*evaluation*.json")):
        frames.append(load_raresim_json(json_path))

    for tsv_path in sorted(raresim_dir.rglob("*summary*.tsv")):
        frames.append(load_raresim_tsv(tsv_path))

    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames, ignore_index=True)

    # Prefer JSON over TSV for same dataset/method/system_type because JSON may contain R@20.
    metrics = (
        metrics.sort_values("source_priority")
        .drop_duplicates(subset=["dataset", "method", "system_type"], keep="first")
        .drop(columns=["source_priority"])
    )

    return metrics


def load_validation_results(validation_dir: Path | None) -> pd.DataFrame:
    """Load validation-tool TSV results."""
    if validation_dir is None or not validation_dir.exists():
        return pd.DataFrame()

    frames = []

    for tsv_path in sorted(validation_dir.rglob("*summary*.tsv")):
        frames.append(load_validation_tsv(tsv_path))

    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames, ignore_index=True)
    return metrics.drop(columns=["source_priority"])


def _filter_datasets(
    metrics: pd.DataFrame,
    allowed_datasets: list[str] | None,
) -> pd.DataFrame:
    """Keep only the requested datasets, warning about anything dropped."""
    if allowed_datasets is None or metrics.empty:
        return metrics

    allowed = set(allowed_datasets)
    present = set(metrics["dataset"].unique())

    missing = [name for name in allowed_datasets if name not in present]
    if missing:
        print(f"[load_results] requested datasets not found on disk: {missing}")

    dropped = present - allowed
    if dropped:
        print(f"[load_results] ignoring datasets outside the report set: {sorted(dropped)}")

    return metrics[metrics["dataset"].isin(allowed)].copy()


def load_all_results(
    raresim_dir: Path | None,
    validation_dir: Path | None,
    allowed_datasets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load RareSim and validation results.

    If allowed_datasets is given, only those datasets are kept (used to restrict
    the report to the PhenoBrain benchmark set).
    """
    raresim = load_raresim_results(raresim_dir)
    validation = load_validation_results(validation_dir)

    has_raresim = not raresim.empty
    has_validation = not validation.empty

    if has_raresim and has_validation:
        metrics = pd.concat([raresim, validation], ignore_index=True)
    elif has_raresim:
        metrics = raresim.copy()
    elif has_validation:
        metrics = validation.copy()
    else:
        raise FileNotFoundError(
            "No results found. Check --raresim and --validation paths."
        )

    numeric_cols = [
        "n_cases",
        "found_count",
        "found_rate",
        "R@1",
        "R@3",
        "R@5",
        "R@10",
        "R@20",
        "MRR",
        "NDCG@10",
        "avg_query_time_sec",
        "median_rank_found",
    ]

    for col in numeric_cols:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    return _filter_datasets(metrics, allowed_datasets)


def compute_case_agreement_from_rank_rows(path: Path, dataset: str) -> pd.DataFrame:
    """Compute hard/easy/unique/consensus cases from a RareSim rank TSV."""
    df = read_tsv(path)

    if "rank" not in df.columns or "case_id" not in df.columns or "method" not in df.columns:
        return pd.DataFrame()

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["found"] = df["rank"].notna()

    methods_per_case = df.groupby("case_id")["method"].nunique()
    found_per_case = df.groupby("case_id")["found"].sum()
    has_rank1 = df.assign(top1=df["rank"].eq(1)).groupby("case_id")["top1"].any()

    rows = []
    for case_id in found_per_case.index:
        rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "n_methods": int(methods_per_case.loc[case_id]),
                "n_methods_found": int(found_per_case.loc[case_id]),
                "hard_case": bool(found_per_case.loc[case_id] == 0),
                "unique_find_case": bool(found_per_case.loc[case_id] == 1),
                "easy_rank1_case": bool(has_rank1.loc[case_id]),
                "consensus_case": bool(
                    found_per_case.loc[case_id] == methods_per_case.loc[case_id]
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_case_agreement_from_json(path: Path) -> pd.DataFrame:
    """Compute case agreement from RareSim JSON rank_matrix."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if "rank_matrix" not in data:
        return pd.DataFrame()

    dataset = normalize_dataset_name(path.parent.name)
    if dataset in {"EVALUATION", "RESULTS", "OUTPUTS"}:
        dataset = normalize_dataset_name(path.name)

    rows = []
    for item in data["rank_matrix"]:
        ranks = item.get("ranks", {})
        values = list(ranks.values())
        n_methods = len(values)
        n_found = sum(value is not None for value in values)
        has_rank1 = any(value == 1 for value in values if value is not None)

        rows.append(
            {
                "dataset": dataset,
                "case_id": f"case_{int(item.get('case_index')):04d}",
                "n_methods": n_methods,
                "n_methods_found": n_found,
                "hard_case": bool(n_found == 0),
                "unique_find_case": bool(n_found == 1),
                "easy_rank1_case": bool(has_rank1),
                "consensus_case": bool(n_found == n_methods),
            }
        )

    return pd.DataFrame(rows)


def load_case_agreement(
    raresim_dir: Path | None,
    allowed_datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Load case-level agreement from RareSim JSON or TSV files."""
    if raresim_dir is None or not raresim_dir.exists():
        return pd.DataFrame()

    frames = []

    for json_path in sorted(raresim_dir.rglob("*evaluation*.json")):
        frame = compute_case_agreement_from_json(json_path)
        if not frame.empty:
            frames.append(frame)

    json_datasets = set()
    if frames:
        json_datasets = set(pd.concat(frames)["dataset"].unique())

    for tsv_path in sorted(raresim_dir.rglob("*summary*.tsv")):
        dataset = normalize_dataset_name(tsv_path.parent.name)
        if dataset in json_datasets:
            continue

        frame = compute_case_agreement_from_rank_rows(tsv_path, dataset)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    case_agreement = pd.concat(frames, ignore_index=True)
    return _filter_datasets(case_agreement, allowed_datasets)
