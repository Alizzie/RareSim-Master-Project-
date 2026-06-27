"""
Create visual answers for RareSim evaluation questions

Q1: Which method performs best?
Q2: Which method ranks the correct disease highest?
Q3: Which method is too slow compared with its performance?
Q4: Are some datasets much harder?
Q5: Do validation tools perform better or worse than RareSim methods?
Q6: Does combining methods (RRF) beat the best single method?
Q7: How do the method families compare overall?

Run from project root:
    python -m scripts.visualizations.benchmark_evaluation.plot_evaluation_questions \
    --raresim outputs/evaluation \
    --validation outputs/validation_tools \
    --output outputs/evaluation_visual_questions
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.visualizations.benchmark_evaluation.config import (
    DATASET_COLORS,
    DATASETS,
    RECALL_COLUMNS,
    SYSTEM_TYPE_COLORS,
)
from scripts.visualizations.benchmark_evaluation.load_results import (
    load_all_results,
    load_case_agreement,
)


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def set_house_style() -> None:
    """One consistent look for every figure."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dddddd",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.6,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def safe_filename(value: str) -> str:
    """Convert a label into a filesystem-safe lowercase filename fragment."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.lower()).strip("_")


def datasets_in_order(metrics: pd.DataFrame) -> list[str]:
    """Datasets present, in the configured display order."""
    present = set(metrics["dataset"].unique())
    return [d for d in DATASETS if d in present]


def ymax_for(values) -> float:
    """Return a padded y-axis maximum for numeric plot values."""
    arr = np.asarray(values, dtype=float)
    top = np.nanmax(arr) if arr.size else 0.0
    return max(0.6, float(top) + 0.06)


def label_system_type(metrics: pd.DataFrame) -> dict[str, str]:
    """Map each method_label to its system type for colouring."""
    grouped = (
        metrics.dropna(subset=["method_label", "system_type"])
        .groupby("method_label")["system_type"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    return {
        str(method_label): str(system_type)
        for method_label, system_type in grouped.items()
    }


def save(fig_path: Path) -> None:
    """Save the current Matplotlib figure and close it."""
    plt.savefig(fig_path)
    plt.close()


# ---------------------------------------------------------------------------
# Q1 — which method performs best?
# ---------------------------------------------------------------------------

# pylint: disable=too-many-locals
def q1_best_method(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of Recall@10 (method x dataset), sorted by mean performance."""
    order_cols = datasets_in_order(metrics)
    pivot = (
        metrics.pivot_table(index="method_label", columns="dataset",
                            values="R@10", aggfunc="max")
        .reindex(columns=order_cols)
    )
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    type_map = label_system_type(metrics)
    n_rows = len(pivot)

    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(order_cols), 1.0 + 0.42 * n_rows))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=ymax_for(data), aspect="auto")

    ax.set_xticks(range(len(order_cols)))
    ax.set_xticklabels(order_cols, rotation=20, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index)

    # Colour each method name by its system type.
    for tick, label in zip(ax.get_yticklabels(), pivot.index):
        tick.set_color(SYSTEM_TYPE_COLORS.get(type_map.get(label, ""), "#333333"))
        tick.set_fontweight("bold")

    for i in range(n_rows):
        for j in range(len(order_cols)):
            val = data[i, j]
            if np.isnan(val):
                continue
            shade = "white" if val > 0.5 * ymax_for(data) else "#1a1a1a"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=shade)

    ax.set_title("Q1 · Which method performs best? (Recall@10)")
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(order_cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Recall@10", rotation=270, labelpad=14)

    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=9,
                      markerfacecolor=c, markeredgecolor=c, label=t)
               for t, c in SYSTEM_TYPE_COLORS.items()]
    ax.legend(handles=handles, title="Method label colour",
              bbox_to_anchor=(0.5, -0.18 - 0.01 * len(order_cols)),
              loc="upper center", ncol=3)

    save(output_dir / "q1_best_method_recall10_heatmap.png")

    # Companion table: best method per criterion per dataset.
    best_rows = []
    for dataset, group in metrics.groupby("dataset"):
        for crit, sort_keys in [
            ("Best Recall@10", ["R@10", "MRR"]),
            ("Best MRR", ["MRR", "R@10"]),
            ("Best Recall@1", ["R@1", "R@10"]),
        ]:
            best = group.sort_values(sort_keys, ascending=False).iloc[0]
            best_rows.append({
                "dataset": dataset,
                "criterion": crit,
                "method_or_tool": best["method_label"],
                "system_type": best["system_type"],
                "value": best[sort_keys[0]],
            })
    pd.DataFrame(best_rows).to_csv(
        output_dir / "q1_best_methods_by_dataset.csv", index=False)


# ---------------------------------------------------------------------------
# Q2 — which method ranks the correct disease highest?
# ---------------------------------------------------------------------------

def q2_recall_curves(metrics: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    """Recall@k curves for the top methods on each dataset."""
    type_map = label_system_type(metrics)
    for dataset in datasets_in_order(metrics):
        subset = (
            metrics[metrics["dataset"] == dataset]
            .sort_values(["R@10", "MRR"], ascending=False)
            .head(top_n)
        )
        if subset.empty:
            continue

        _, ax = plt.subplots(figsize=(10, 6))
        ks = [int(c.split("@")[1]) for c in RECALL_COLUMNS]
        for _, row in subset.iterrows():
            ys = [row[c] for c in RECALL_COLUMNS]
            color = SYSTEM_TYPE_COLORS.get(type_map.get(row["method_label"], ""), "#888888")
            ls = "--" if row["system_type"] == "Validation tool" else "-"
            ax.plot(ks, ys, marker="o", linewidth=2, linestyle=ls,
                    color=color, label=row["method_label"])

        ax.set_xticks(ks)
        ax.set_xticklabels([f"@{k}" for k in ks])
        ax.set_title(f"Q2 · {dataset} — how high is the correct disease ranked?")
        ax.set_xlabel("Recall cut-off k")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, ymax_for(subset[RECALL_COLUMNS].to_numpy()))
        ax.legend(title="Method / tool", bbox_to_anchor=(1.02, 1), loc="upper left")
        save(output_dir / f"q2_{safe_filename(dataset)}_recall_curve_top_methods.png")


# ---------------------------------------------------------------------------
# Q3 — which method is too slow for its performance?
# ---------------------------------------------------------------------------

def q3_speed_vs_performance(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Scatter average runtime against Recall@10 for each dataset."""
    for dataset in datasets_in_order(metrics):
        subset = metrics[
            (metrics["dataset"] == dataset)
            & metrics["avg_query_time_sec"].notna()
            & (metrics["avg_query_time_sec"] > 0)
        ].copy()
        if subset.empty:
            continue

        _, ax = plt.subplots(figsize=(10.5, 6.5))
        for stype, grp in subset.groupby("system_type"):
            stype_str = str(stype)

            ax.scatter(
                grp["avg_query_time_sec"],
                grp["R@10"],
                s=90,
                color=SYSTEM_TYPE_COLORS.get(stype_str, "#888888"),
                edgecolor="white",
                linewidth=1.2,
                label=stype_str,
                zorder=3,
            )
        for _, row in subset.iterrows():
            ax.annotate(row["method_label"],
                        (row["avg_query_time_sec"], row["R@10"]),
                        fontsize=8.5, xytext=(5, 5), textcoords="offset points",
                        color="#444444")

        ax.set_xscale("log")
        ax.set_title(f"Q3 · {dataset} — speed vs Recall@10")
        ax.set_xlabel("Average time per case (s, log scale) — left is faster")
        ax.set_ylabel("Recall@10 — higher is better")
        ax.set_ylim(0, ymax_for(subset["R@10"]))
        ax.legend(title="System type", loc="lower right")
        ax.text(0.01, 0.98, "↖ fast & accurate = best", transform=ax.transAxes,
                fontsize=9, color="#2a7f2a", va="top")
        save(output_dir / f"q3_{safe_filename(dataset)}_speed_vs_recall10.png")


# ---------------------------------------------------------------------------
# Q4 — are some datasets much harder?
# ---------------------------------------------------------------------------

def q4_dataset_difficulty(metrics: pd.DataFrame,
                          case_agreement: pd.DataFrame,
                          output_dir: Path) -> None:
    """Summarise dataset difficulty using best and average Recall@10."""
    order = datasets_in_order(metrics)

    summary = (
        metrics.groupby("dataset")
        .agg(best_recall10=("R@10", "max"),
             mean_recall10=("R@10", "mean"),
             best_mrr=("MRR", "max"))
        .reindex(order)
        .reset_index()
    )
    if not case_agreement.empty:
        hard = (case_agreement.groupby("dataset")["hard_case"].mean()
                .reindex(order).reset_index(name="hard_case_rate"))
        summary = summary.merge(hard, on="dataset", how="left")
    summary.to_csv(output_dir / "q4_dataset_difficulty_summary.csv", index=False)

    _, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(order))
    colors = [DATASET_COLORS.get(d, "#888888") for d in order]
    ax.bar(x - 0.2, summary["best_recall10"], width=0.4, color=colors,
           label="Best Recall@10 (any system)")
    ax.bar(x + 0.2, summary["mean_recall10"], width=0.4, color=colors, alpha=0.45,
           label="Mean Recall@10 (across methods)")
    for xi, (b, m) in enumerate(zip(summary["best_recall10"], summary["mean_recall10"])):
        ax.text(xi - 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, m + 0.01, f"{m:.2f}", ha="center", fontsize=9, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_title("Q4 · Are some datasets harder? (lower bars = harder)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, ymax_for(summary["best_recall10"]))
    ax.legend(loc="upper right")
    save(output_dir / "q4_dataset_difficulty.png")

    if not case_agreement.empty:
        case_agreement.to_csv(output_dir / "case_agreement_by_case.csv", index=False)


# ---------------------------------------------------------------------------
# Q5 — validation tools vs RareSim
# ---------------------------------------------------------------------------

# pylint: disable=too-many-locals
def q5_validation_vs_raresim(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare the best validation tools with the best RareSim systems."""
    order = datasets_in_order(metrics)
    best_rows = []
    for (dataset, stype), grp in metrics.groupby(["dataset", "system_type"]):
        if grp["R@10"].notna().sum() == 0:
            continue
        best = grp.sort_values(["R@10", "MRR"], ascending=False).iloc[0]
        best_rows.append({
            "dataset": dataset, "system_type": stype,
            "best_method_or_tool": best["method_label"],
            "best_recall10": best["R@10"], "best_mrr": best["MRR"],
        })
    best_by_system = pd.DataFrame(best_rows)
    best_by_system.to_csv(output_dir / "q5_best_by_system_type.csv", index=False)
    if best_by_system.empty:
        return

    pivot = (best_by_system.pivot(index="dataset", columns="system_type",
                                  values="best_recall10").reindex(order))
    type_cols = [c for c in ["RareSim method", "Ensemble", "Validation tool"]
                 if c in pivot.columns]

    _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot))
    width = 0.8 / max(1, len(type_cols))
    for i, col in enumerate(type_cols):
        ax.bar(x + (i - (len(type_cols) - 1) / 2) * width, pivot[col], width,
               color=SYSTEM_TYPE_COLORS.get(col, "#888"), label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_title("Q5 · Best Recall@10 — validation tools vs RareSim")
    ax.set_ylabel("Best Recall@10")
    ax.set_ylim(0, ymax_for(pivot.to_numpy()))
    ax.legend(title="System type", loc="upper right")
    save(output_dir / "q5_validation_vs_raresim_best_recall10.png")

    # Difference: validation best minus RareSim best (methods + ensembles).
    diff = pd.DataFrame(index=pivot.index)
    raresim_cols = [c for c in ["RareSim method", "Ensemble"] if c in pivot.columns]
    if "Validation tool" in pivot.columns and raresim_cols:
        diff["delta"] = pivot["Validation tool"] - pivot[raresim_cols].max(axis=1)
        diff = diff.reset_index()
        diff.to_csv(output_dir / "q5_difference_table.csv", index=False)

        _, ax = plt.subplots(figsize=(10, 5.5))
        colors = ["#d1603d" if v > 0 else "#2a7f9e" for v in diff["delta"]]
        ax.bar(diff["dataset"], diff["delta"], color=colors)
        ax.axhline(0, color="#333", linewidth=1)
        for xi, v in enumerate(diff["delta"]):
            ax.text(xi, v + (0.005 if v >= 0 else -0.02), f"{v:+.2f}",
                    ha="center", fontsize=9)
        ax.set_title("Q5 · Validation best − RareSim best (Recall@10)")
        ax.set_ylabel("Δ Recall@10")
        ax.set_xticks(range(len(diff)))
        ax.set_xticklabels(diff["dataset"], rotation=15, ha="right")
        ax.text(0.99, 0.95, "above 0: validation wins\nbelow 0: RareSim wins",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")
        save(output_dir / "q5_validation_minus_raresim_difference.png")


# ---------------------------------------------------------------------------
# Q6 — does combining methods (RRF) beat the best single method?
# ---------------------------------------------------------------------------

# pylint: disable=too-many-locals
def q6_ensemble_gain(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare the best RRF ensemble with the best single RareSim method."""
    order = datasets_in_order(metrics)
    rows = []
    for dataset in order:
        grp = metrics[metrics["dataset"] == dataset]
        single = grp[grp["system_type"] == "RareSim method"]
        ens = grp[grp["system_type"] == "Ensemble"]
        if single.empty or ens.empty:
            continue
        b_single = single.sort_values("R@10", ascending=False).iloc[0]
        b_ens = ens.sort_values("R@10", ascending=False).iloc[0]
        rows.append({
            "dataset": dataset,
            "best_single_method": b_single["method_label"],
            "best_single_recall10": b_single["R@10"],
            "best_ensemble": b_ens["method_label"],
            "best_ensemble_recall10": b_ens["R@10"],
            "gain": b_ens["R@10"] - b_single["R@10"],
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "q6_ensemble_gain.csv", index=False)

    _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["best_single_recall10"], 0.4,
           color=SYSTEM_TYPE_COLORS["RareSim method"], label="Best single method")
    ax.bar(x + 0.2, df["best_ensemble_recall10"], 0.4,
           color=SYSTEM_TYPE_COLORS["Ensemble"], label="Best RRF ensemble")
    gains = pd.to_numeric(df["gain"], errors="coerce").to_numpy(dtype=float)
    ensemble_recall10s = pd.to_numeric(
        df["best_ensemble_recall10"],
        errors="coerce",
    ).to_numpy(dtype=float)

    for xi, (gain, ensemble_recall10) in enumerate(zip(gains, ensemble_recall10s)):
        ax.text(
            xi + 0.2,
            ensemble_recall10 + 0.01,
            f"{gain:+.2f}",
            ha="center",
            fontsize=9,
            color="#2a7f2a" if gain >= 0 else "#b23",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=15, ha="right")
    ax.set_title("Q6 · Does fusion help? RRF ensemble vs best single method")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, ymax_for(df[["best_single_recall10", "best_ensemble_recall10"]].to_numpy()))
    ax.legend(loc="upper right")
    save(output_dir / "q6_ensemble_vs_single.png")


# ---------------------------------------------------------------------------
# Q7 — method family overview
# ---------------------------------------------------------------------------

def q7_family_overview(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean Recall@10 for each method family across datasets."""
    family = (
        metrics.groupby("method_family")["R@10"]
        .mean()
        .sort_values(ascending=True)
    )

    if family.empty:
        return

    family.to_frame("mean_recall10").to_csv(output_dir / "q7_family_overview.csv")

    family_labels = [str(label) for label in family.index]
    family_values = pd.to_numeric(family, errors="coerce").to_numpy(dtype=float)

    _, ax = plt.subplots(figsize=(10, 0.55 * len(family_labels) + 1.5))

    colors = [
        "#d1603d" if label == "Validation tool"
        else "#6a4c93" if label == "Ensemble"
        else "#2a7f9e"
        for label in family_labels
    ]

    ax.barh(family_labels, family_values, color=colors)

    for i, value in enumerate(family_values):
        ax.text(
            value + 0.005,
            i,
            f"{value:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_title("Q7 · Mean Recall@10 by method family (across datasets)")
    ax.set_xlabel("Mean Recall@10")
    ax.set_xlim(0, ymax_for(family_values))
    save(output_dir / "q7_family_overview.png")


# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, load metrics, and generate all evaluation plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raresim", type=Path, default=Path("evaluation"))
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/evaluation_visual_questions"))
    parser.add_argument("--top-n", type=int, default=7,
                        help="Methods shown in Q2 recall-curve plots.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    set_house_style()

    metrics = load_all_results(args.raresim, args.validation,
                               allowed_datasets=DATASETS)
    case_agreement = load_case_agreement(args.raresim,
                                         allowed_datasets=DATASETS)

    metrics.to_csv(args.output / "combined_metrics.csv", index=False)

    q1_best_method(metrics, args.output)
    q2_recall_curves(metrics, args.output, top_n=args.top_n)
    q3_speed_vs_performance(metrics, args.output)
    q4_dataset_difficulty(metrics, case_agreement, args.output)
    q5_validation_vs_raresim(metrics, args.output)
    q6_ensemble_gain(metrics, args.output)
    q7_family_overview(metrics, args.output)

    print(f"Saved plots and tables to: {args.output}")


if __name__ == "__main__":
    main()
