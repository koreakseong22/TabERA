"""Create paper-ready tables and figures for the explanation analysis.

The input is the dataset-level CSV produced by
``aggregate_explanation_analysis.py``.  This script never modifies the
audited aggregate files; it writes descriptive tables and figures to a
separate output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTLIER_DATASET = 1493

TABLE2_METRICS = [
    "global_majority_acc",
    "region_majority_acc",
    "regional_baseline_acc",
    "final_acc",
    "prediction_change_rate",
    "corrected_rate",
    "degraded_rate",
]

TABLE3_METRICS = [
    "same_region_share",
    "fallback_rate",
    "global_knn_jaccard",
    "global_knn_overlap_coverage",
    "label_agreement_gain",
    "label_gain_eligible_coverage",
    "label_agreement_delta_vs_global",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-summary",
        default="analysis_results/aggregate/dataset_summary.csv",
    )
    parser.add_argument("--dataset-metadata", default="dataset_id.json")
    parser.add_argument("--output", default="analysis_results/paper_outputs")
    return parser.parse_args()


def dataset_labels(frame: pd.DataFrame, metadata_path: Path) -> dict[int, str]:
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = {}
    for dataset_id in frame["dataset_id"].astype(int):
        item = metadata.get(str(dataset_id), {})
        name = item.get("fullname") or item.get("name")
        labels[dataset_id] = f"{dataset_id} ({name})" if name else str(dataset_id)
    return labels


def descriptive_table(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    without_outlier = frame.loc[frame["dataset_id"] != OUTLIER_DATASET]
    rows = []
    for metric in metrics:
        values = frame[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "dataset_equal_mean": values.mean(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "mean_excluding_openml_1493": without_outlier[metric].mean(),
                "defined_datasets": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def win_tie_loss(delta: pd.Series, tolerance: float = 1e-12) -> dict[str, int]:
    values = delta.to_numpy(dtype=float)
    return {
        "wins": int(np.sum(values > tolerance)),
        "ties": int(np.sum(np.abs(values) <= tolerance)),
        "losses": int(np.sum(values < -tolerance)),
    }


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_gain(frame: pd.DataFrame, labels: dict[int, str], output: Path) -> None:
    plot = frame[["dataset_id", "global_majority_acc", "region_majority_acc"]].copy()
    plot["gain"] = plot["region_majority_acc"] - plot["global_majority_acc"]
    plot = plot.sort_values("gain")
    y = np.arange(len(plot))
    colors = np.where(plot["gain"] >= 0, "#2878B5", "#C44E52")

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.hlines(y, 0, 100 * plot["gain"], color=colors, linewidth=2)
    ax.scatter(100 * plot["gain"], y, color=colors, s=32, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, [labels[int(i)] for i in plot["dataset_id"]], fontsize=8)
    ax.set_xlabel("Region-majority minus global-majority accuracy (percentage points)")
    ax.set_ylabel("OpenML dataset")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output / "region_majority_gain_by_dataset")


def correction_panels(
    frame: pd.DataFrame,
    labels: dict[int, str],
    stem: Path,
    annotate_outlier: bool,
) -> None:
    plot = frame.copy()
    plot["accuracy_delta"] = plot["final_acc"] - plot["regional_baseline_acc"]
    plot = plot.sort_values("prediction_change_rate")
    y = np.arange(len(plot))
    names = [labels[int(i)] for i in plot["dataset_id"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), sharey=True)
    axes[0].barh(y, 100 * plot["prediction_change_rate"], color="#4C956C")
    axes[0].set_xlabel("Prediction-change rate (%)")
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_ylabel("OpenML dataset")

    delta = 100 * plot["accuracy_delta"]
    colors = np.where(delta >= 0, "#2878B5", "#C44E52")
    axes[1].barh(y, delta, color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("Final minus regional-baseline accuracy (percentage points)")

    for ax in axes:
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines[["top", "right", "left"]].set_visible(False)

    row = plot.reset_index(drop=True).query("dataset_id == @OUTLIER_DATASET")
    if annotate_outlier and not row.empty:
        pos = int(row.index[0])
        axes[0].annotate(
            "OpenML 1493",
            (100 * row.iloc[0]["prediction_change_rate"], pos),
            xytext=(-6, 7),
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )
        axes[1].annotate(
            "OpenML 1493",
            (100 * row.iloc[0]["accuracy_delta"], pos),
            xytext=(-6, 7),
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, stem)


def plot_correction(frame: pd.DataFrame, labels: dict[int, str], output: Path) -> None:
    correction_panels(
        frame,
        labels,
        output / "correction_behavior_by_dataset",
        annotate_outlier=True,
    )
    correction_panels(
        frame.loc[frame["dataset_id"] != OUTLIER_DATASET],
        labels,
        output / "correction_behavior_excluding_openml_1493",
        annotate_outlier=False,
    )


def main() -> int:
    args = parse_args()
    source = Path(args.dataset_summary).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if len(frame) != 21 or frame["dataset_id"].nunique() != 21:
        raise ValueError("Expected exactly 21 dataset-level rows")
    required = set(TABLE2_METRICS + TABLE3_METRICS + ["dataset_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    labels = dataset_labels(frame, Path(args.dataset_metadata))
    descriptive_table(frame, TABLE2_METRICS).to_csv(
        output / "table2_descriptive.csv", index=False
    )
    descriptive_table(frame, TABLE3_METRICS).to_csv(
        output / "table3_descriptive.csv", index=False
    )

    region_delta = frame["region_majority_acc"] - frame["global_majority_acc"]
    correction_delta = frame["final_acc"] - frame["regional_baseline_acc"]
    sensitivity = {
        "source": str(source),
        "n_datasets": int(len(frame)),
        "outlier_dataset": OUTLIER_DATASET,
        "region_majority_minus_global_majority": {
            "mean": float(region_delta.mean()),
            "median": float(region_delta.median()),
            "mean_excluding_openml_1493": float(
                region_delta[frame["dataset_id"] != OUTLIER_DATASET].mean()
            ),
            **win_tie_loss(region_delta),
        },
        "final_minus_regional_baseline": {
            "mean": float(correction_delta.mean()),
            "median": float(correction_delta.median()),
            "mean_excluding_openml_1493": float(
                correction_delta[frame["dataset_id"] != OUTLIER_DATASET].mean()
            ),
            **win_tie_loss(correction_delta),
        },
        "label_agreement_gain": {
            "mean": float(frame["label_agreement_gain"].mean()),
            "median": float(frame["label_agreement_gain"].median()),
            "mean_excluding_openml_1493": float(
                frame.loc[
                    frame["dataset_id"] != OUTLIER_DATASET, "label_agreement_gain"
                ].mean()
            ),
            **win_tie_loss(frame["label_agreement_gain"]),
        },
    }
    (output / "sensitivity_summary.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )

    plot_region_gain(frame, labels, output)
    plot_correction(frame, labels, output)
    print(f"Wrote paper outputs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
