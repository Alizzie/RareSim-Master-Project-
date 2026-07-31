"""Standardize local PhenoBrain raw results into the benchmark TSV format.

Prerequisite: run PhenoBrain locally first (see the "PhenoBrain (Local)" wiki
page) to produce the raw per-model rank export (.xlsx). This script does NOT
run PhenoBrain — it only converts that raw export into standardized
per-dataset summaries consumed by compare_methods.py, remapping `RD:` disease
codes to OMIM/ORPHA via the same disease-list-detail API as run_phenobrain.py.
Query times are not available in the raw export and are recorded as None.

By default every model rank column in the raw export is standardized in one
pass, each written to its own `phenobrain (<model>)_benchmarks/` folder — so
compare_methods.py's auto-discovery treats each model as its own method,
with no need to name models individually.

Usage:
    # Standardize ALL model columns found in the raw export (default behavior)
    python3 run_phenobrain_local.py --input phenobrain_raw/0.1.27.xlsx --dataset 0.1.27

    # Restrict to one or more specific model columns
    python3 run_phenobrain_local.py \\
        --input phenobrain_raw/0.1.27.xlsx --dataset 0.1.27 \\
        --models NN-Mixup-Random-1 MICA-QD-Random

    # List the model columns available in a raw export, then exit
    python3 run_phenobrain_local.py --input phenobrain_raw/0.1.27.xlsx --list-models

    # Skip the RD -> OMIM/ORPHA API remap (offline; keep raw RD codes)
    python3 run_phenobrain_local.py \\
        --input phenobrain_raw/0.1.27.xlsx --dataset 0.1.27 --no-remap

    # A raw CSV/TSV export also still works (delimiter auto-detected)
    python3 run_phenobrain_local.py --input phenobrain_raw/0.1.27.csv --dataset 0.1.27
"""

import ast
import csv
import argparse
from pathlib import Path

from _utils import save_summary_tsv, compute_stats, print_stats

# Reuse the exact RD -> OMIM/ORPHA remapping used by the hosted runner.
from run_phenobrain import create_RD_code_mapper
from raresim.utils.paths import OUTPUTS_DIR

VAL_OUTPUTS_DIR = OUTPUTS_DIR / "validation_tools"

# Everything except these header columns is treated as a per-model rank column.
FIXED_COLUMNS = ["DATA_RANK", "DISEASE_CODE", "DISEASE_NAME", "HPO_CODE", "HPO_NAME"]


def out_dir_for_model(base_dir: Path, model: str) -> Path:
    """Each model gets its own benchmark folder so compare_methods.py's
    auto-discovery (folder name minus '_benchmarks' = method name) treats
    every model as a separate method, e.g. 'phenobrain (NN-Mixup-Random-1)'."""
    return base_dir / f"phenobrain ({model})_benchmarks"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Standardize local PhenoBrain raw results (.xlsx/.csv/.tsv).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw PhenoBrain export: .xlsx (default local export "
        "format), or .csv/.tsv.",
    )
    p.add_argument(
        "--dataset",
        help="Dataset name (used for case_id prefix and output filename). "
        "Required unless --list-models is set.",
    )
    # NOTE: no single --model flag — see --models above; default is "all".
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model rank columns to standardize (e.g. "
        "'NN-Mixup-Random-1'). Default: every model column found in --input. "
        "See --list-models for the options.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=VAL_OUTPUTS_DIR,
        help=f"Base output directory; each model gets its own "
        f"'phenobrain (<model>)_benchmarks' subfolder here (default: {VAL_OUTPUTS_DIR})",
    )
    p.add_argument(
        "--delimiter",
        default=None,
        help="Column delimiter for CSV/TSV input (ignored for .xlsx). "
        "Auto-detected (tab vs comma) when omitted.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Treat a ground-truth rank greater than this as not found "
        "(default: keep all ranks).",
    )
    p.add_argument(
        "--no-remap",
        action="store_true",
        help="Skip the RD->OMIM/ORPHA API remap and keep raw RD codes (offline mode).",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model columns found in --input and exit.",
    )
    return p.parse_args()


# ── Parsing helpers ───────────────────────────────────────────────────────────
def read_rows_xlsx(path: Path) -> tuple[list[str], list[dict]]:
    """Read the raw PhenoBrain .xlsx export. Uses the first sheet; all cell
    values are stringified so downstream parsing matches the CSV/TSV path."""
    try:
        import openpyxl
    except ImportError as e:
        raise SystemExit(
            "Reading .xlsx requires openpyxl: "
            "pip install openpyxl --break-system-packages"
        ) from e

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        raise SystemExit("The Excel workbook does not contain any worksheets.")

    rows_iter = ws.iter_rows(values_only=True)

    header = next(rows_iter, None) or ()
    fieldnames = [str(h).strip() if h is not None else "" for h in header]

    rows = []
    for raw_row in rows_iter:
        if raw_row is None or all(v is None for v in raw_row):
            continue
        row = {}
        for key, val in zip(fieldnames, raw_row):
            if not key:
                continue
            row[key] = "" if val is None else str(val)
        rows.append(row)

    wb.close()
    return fieldnames, rows


def read_rows_delimited(
    path: Path, delimiter: str | None
) -> tuple[list[str], list[dict]]:
    """Read a raw CSV/TSV export. Auto-detects tab vs comma when delimiter is None."""
    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
        f.seek(0)
        delim = delimiter or ("\t" if "\t" in first_line else ",")
        reader = csv.DictReader(f, delimiter=delim)
        return list(reader.fieldnames or []), list(reader)


def read_rows(path: Path, delimiter: str | None) -> tuple[list[str], list[dict]]:
    """Read the raw PhenoBrain export, dispatching on file extension:
    .xlsx/.xlsm via openpyxl, everything else as delimited CSV/TSV."""
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return read_rows_xlsx(path)
    return read_rows_delimited(path, delimiter)


def model_columns(fieldnames: list[str]) -> list[str]:
    """Return the model rank columns (everything that is not a fixed column)."""
    return [c for c in fieldnames if c and c not in FIXED_COLUMNS]


def parse_list_cell(value: str) -> list[str]:
    """Parse a stringified Python list cell, e.g. "['RD:6786', 'RD:2700']"."""
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value]
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


def rank_from_cell(value: str, topk: int | None) -> int | None:
    """Convert a rank cell to int. Returns None if empty, non-numeric, or > topk."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        rank = int(float(value))
    except ValueError:
        return None
    if topk is not None and rank > topk:
        return None
    return rank


# ── RD code remapping ─────────────────────────────────────────────────────────
def build_code_mapper(rows: list[dict], batch_size: int = 500) -> dict:
    """Collect all ground-truth RD codes and remap them in batches via the API."""
    codes: set[str] = set()
    for row in rows:
        codes.update(parse_list_cell(row.get("DISEASE_CODE", "")))
    ordered = sorted(c for c in codes if c)

    mapper: dict = {}
    for i in range(0, len(ordered), batch_size):
        mapper.update(create_RD_code_mapper(ordered[i : i + batch_size]))
    return mapper


# ── Standardization ───────────────────────────────────────────────────────────
def standardize(
    rows: list[dict],
    dataset: str,
    model: str,
    code_mapper: dict | None,
    topk: int | None,
) -> list[dict]:
    """Convert raw rows into the standardized summary format (one row per case)."""
    dataset_key = dataset.lower()
    summary = []

    for i, row in enumerate(rows):
        raw_rank = (row.get("DATA_RANK") or "").strip()
        try:
            idx = int(float(raw_rank))
        except ValueError:
            idx = i
        case_id = f"{dataset_key}_case_{idx:04d}"

        gt_rd = parse_list_cell(row.get("DISEASE_CODE", ""))
        hpo_ids = parse_list_cell(row.get("HPO_CODE", ""))

        if code_mapper is not None:
            mapped = [code_mapper.get(c) for c in gt_rd]
            mapped = [m for m in mapped if m]
            confirmed = ";".join(mapped) if mapped else ";".join(gt_rd)
        else:
            confirmed = ";".join(gt_rd)

        rank = rank_from_cell(row.get(model, ""), topk)
        found = rank is not None

        summary.append(
            {
                "case_id": case_id,
                "n_hpo": len(hpo_ids),
                "confirmed_diseases": confirmed if confirmed else "None",
                "rank": rank if found else "None",
                "matched_id": (confirmed if confirmed else "None") if found else "None",
                "score": "None",  # raw export stores ranks, not scores
                "status": True,
                "query_time_sec": "None",  # not available from the raw export
            }
        )

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    fieldnames, rows = read_rows(args.input, args.delimiter)
    models = model_columns(fieldnames)

    if args.list_models:
        print(f"Model columns in {args.input} ({len(models)} found):")
        for m in models:
            print(f"  - {m}")
        return

    if not args.dataset:
        raise SystemExit("--dataset is required (unless --list-models is set).")

    if args.models:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            raise SystemExit(
                f"Model(s) not found in {args.input}: {', '.join(unknown)}\n"
                f"  Available: {', '.join(models)}"
            )
        target_models = args.models
    else:
        target_models = models

    if not target_models:
        raise SystemExit(f"No model columns found in {args.input}.")

    print(f"Loaded {len(rows)} cases from {args.input} ({len(target_models)} model(s))")

    code_mapper = None
    if not args.no_remap:
        code_mapper = build_code_mapper(rows)
        print(f"Remapped {len(code_mapper)} RD codes to OMIM/ORPHA")

    dataset_key = args.dataset.lower()
    for model in target_models:
        summary = standardize(rows, args.dataset, model, code_mapper, args.topk)

        out_dir = out_dir_for_model(args.out_dir, model)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_summary_tsv(summary, out_dir / f"{dataset_key}_summary.tsv")

        stats_path = out_dir / f"{dataset_key}_stats.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            print_stats(dataset_key, compute_stats(summary), f)

        print(f"  [{model}] wrote summary + stats for '{dataset_key}' to {out_dir}")

    print(
        f"Done: {len(target_models)} model(s) standardized for dataset '{dataset_key}'."
    )


if __name__ == "__main__":
    main()
