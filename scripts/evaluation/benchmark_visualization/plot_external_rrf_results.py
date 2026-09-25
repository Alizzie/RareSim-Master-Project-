"""Plot results from the exploratory external-augmented RRF experiment.

Input:
    outputs/ensemble_external_experiment/external_rrf_results.json

Outputs:
    outputs/ensemble_external_experiment/figures/
        external_rrf_recall10_comparison.pdf
        external_rrf_recall10_comparison.png
        external_rrf_summary.tsv

Experiment conditions:
    1. Best individual Rarefully method
    2. Rarefully ensemble (RRF)
    3. Selected external system
    4. A) All Rarefully methods + external system
    5. B) Best Rarefully method + external system
"""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from raresim.utils.paths import OUTPUTS_DIR


EXPERIMENT_DIR = OUTPUTS_DIR / "ensemble_external_experiment"

INPUT_FILE = EXPERIMENT_DIR / "external_rrf_results.json"

FIGURE_DIR = EXPERIMENT_DIR / "figures"


DATASET_ORDER = [
    "MME",
    "HMS",
    "PUMCH_L",
    "PUMCH-ADM",
    "RAMEDIS",
    "LIRICAL",
]


DISPLAY_NAMES = {
    "MME": "MME",
    "HMS": "HMS",
    "PUMCH_L": "PUMCH-L",
    "PUMCH-ADM": "PUMCH-ADM",
    "RAMEDIS": "RAMEDIS",
    "LIRICAL": "LIRICAL",
}


EXTERNAL_DISPLAY_NAMES = {
    "phenobrain": "PhenoBrain",
    "lirical": "LIRICAL",
}


REQUIRED_CONDITIONS = [
    "best_rarefully",
    "rarefully_rrf",
    "external",
    "all_rarefully_plus_external_rrf",
    "best_rarefully_plus_external_rrf",
]


def load_results() -> list[dict]:
    """Load and validate completed external-RRF experiment results."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{INPUT_FILE}\n\n"
            "Run external_rrf_experiment.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        raw_results = json.load(file)

    if not raw_results:
        raise RuntimeError(
            f"No experiment results found in {INPUT_FILE}"
        )

    valid_results = []

    for result in raw_results:
        dataset = result.get("dataset")

        if not dataset:
            print("[warning] Skipping result without dataset name.")
            continue

        missing = [
            condition
            for condition in REQUIRED_CONDITIONS
            if condition not in result
        ]

        if missing:
            print(
                f"[warning] Skipping {dataset}: "
                f"missing {', '.join(missing)}"
            )
            continue

        valid_results.append(result)

    if not valid_results:
        raise RuntimeError(
            "No complete experiment results were found."
        )

    order = {
        dataset: index
        for index, dataset in enumerate(DATASET_ORDER)
    }

    valid_results.sort(
        key=lambda result: order.get(
            str(result["dataset"]),
            len(DATASET_ORDER),
        )
    )

    return valid_results


def metric(
    result: dict,
    condition: str,
    name: str = "recall_10",
) -> float:
    """Read one metric value from one experiment condition."""
    return float(result[condition][name])


def external_name(result: dict) -> str:
    """Return a readable external-system name."""
    system = str(
        result.get("external_system") or "external"
    )

    return EXTERNAL_DISPLAY_NAMES.get(
        system,
        system,
    )


def dataset_name(result: dict) -> str:
    """Return a readable dataset name."""
    dataset = str(
        result.get("dataset") or "Unknown"
    )

    return DISPLAY_NAMES.get(
        dataset,
        dataset,
    )


def write_summary_tsv(
    results: list[dict],
) -> None:
    """Write exact Recall@10 values used in the figure."""
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        FIGURE_DIR
        / "external_rrf_summary.tsv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="\t",
        )

        writer.writerow(
            [
                "dataset",
                "n_cases",
                "best_rarefully_method",
                "external_system",
                "best_rarefully_recall10",
                "rarefully_ensemble_rrf_recall10",
                "selected_external_recall10",
                "A_all_rarefully_plus_external_recall10",
                "B_best_rarefully_plus_external_recall10",
            ]
        )

        for result in results:
            best = metric(
                result,
                "best_rarefully",
            )

            rarefully_rrf = metric(
                result,
                "rarefully_rrf",
            )

            external = metric(
                result,
                "external",
            )

            all_plus = metric(
                result,
                "all_rarefully_plus_external_rrf",
            )

            best_plus = metric(
                result,
                "best_rarefully_plus_external_rrf",
            )

            writer.writerow(
                [
                    result["dataset"],
                    result["n_cases"],
                    result["best_rarefully_method"],
                    external_name(result),
                    f"{best:.4f}",
                    f"{rarefully_rrf:.4f}",
                    f"{external:.4f}",
                    f"{all_plus:.4f}",
                    f"{best_plus:.4f}",
                ]
            )

    print(f"Saved: {path}")


def add_bar_labels(
    ax,
    bars,
) -> None:
    """Add numeric Recall@10 values above bars."""
    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )


def plot_recall10_comparison(
    results: list[dict],
) -> None:
    """Grouped Recall@10 comparison across experiment conditions."""
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets: list[str] = [
        dataset_name(result)
        for result in results
    ]

    best_rarefully = [
        metric(
            result,
            "best_rarefully",
        )
        for result in results
    ]

    rarefully_rrf = [
        metric(
            result,
            "rarefully_rrf",
        )
        for result in results
    ]

    external = [
        metric(
            result,
            "external",
        )
        for result in results
    ]

    all_plus_external = [
        metric(
            result,
            "all_rarefully_plus_external_rrf",
        )
        for result in results
    ]

    best_plus_external = [
        metric(
            result,
            "best_rarefully_plus_external_rrf",
        )
        for result in results
    ]

    x = np.arange(len(datasets))
    width = 0.16

    fig, ax = plt.subplots(
        figsize=(13, 6.5)
    )

    bars_best = ax.bar(
        x - 2 * width,
        best_rarefully,
        width,
        label="Best Rarefully method",
    )

    bars_rrf = ax.bar(
        x - width,
        rarefully_rrf,
        width,
        label="Rarefully ensemble (RRF)",
    )

    bars_external = ax.bar(
        x,
        external,
        width,
        label="Selected external system",
    )

    bars_all_plus = ax.bar(
        x + width,
        all_plus_external,
        width,
        label="A: All Rarefully + external",
    )

    bars_best_plus = ax.bar(
        x + 2 * width,
        best_plus_external,
        width,
        label="B: Best Rarefully + external",
    )

    # Add Recall@10 values above every bar
    add_bar_labels(
        ax,
        bars_best,
    )

    add_bar_labels(
        ax,
        bars_rrf,
    )

    add_bar_labels(
        ax,
        bars_external,
    )

    add_bar_labels(
        ax,
        bars_all_plus,
    )

    add_bar_labels(
        ax,
        bars_best_plus,
    )

    ax.set_ylabel(
        "Recall@10"
    )

    ax.set_xlabel(
        "Dataset"
    )

    ax.set_title(
        "Exploratory External-Augmented Ensemble Performance"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        datasets
    )

    ax.set_ylim(
        0,
        1.06,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    pdf_path = (
        FIGURE_DIR
        / "external_rrf_recall10_comparison.pdf"
    )

    png_path = (
        FIGURE_DIR
        / "external_rrf_recall10_comparison.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def print_results_summary(
    results: list[dict],
) -> None:
    """Print a compact human-readable result table."""
    print("\nExternal-augmented RRF results")
    print("=" * 105)

    print(
        f"{'Dataset':<12}"
        f"{'Best RF':>10}"
        f"{'RF RRF':>10}"
        f"{'External':>10}"
        f"{'A':>10}"
        f"{'B':>10}"
        f"  {'External system':<15}"
    )

    print("-" * 105)

    for result in results:
        best = metric(
            result,
            "best_rarefully",
        )

        rarefully_rrf = metric(
            result,
            "rarefully_rrf",
        )

        external = metric(
            result,
            "external",
        )

        all_plus = metric(
            result,
            "all_rarefully_plus_external_rrf",
        )

        best_plus = metric(
            result,
            "best_rarefully_plus_external_rrf",
        )

        print(
            f"{dataset_name(result):<12}"
            f"{best:>10.3f}"
            f"{rarefully_rrf:>10.3f}"
            f"{external:>10.3f}"
            f"{all_plus:>10.3f}"
            f"{best_plus:>10.3f}"
            f"  {external_name(result):<15}"
        )

    print("=" * 105)


def main() -> None:
    results = load_results()

    print(
        f"Loaded {len(results)} completed dataset result(s) "
        f"from {INPUT_FILE}"
    )

    print_results_summary(
        results
    )

    write_summary_tsv(
        results
    )

    plot_recall10_comparison(
        results
    )


if __name__ == "__main__":
    main()
