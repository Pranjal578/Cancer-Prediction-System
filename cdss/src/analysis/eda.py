"""Generate reproducible exploratory data-analysis outputs for WDBC."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_eda_outputs(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> dict[str, Path]:
    """Save class, distribution, correlation, and summary analyses for WDBC."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs: dict[str, Path] = {}

    summary_path = output_dir / "feature_summary.csv"
    X.describe().T.assign(missing_values=X.isna().sum()).to_csv(summary_path)
    outputs["feature_summary"] = summary_path

    correlation_path = output_dir / "feature_correlations.csv"
    X.corr(numeric_only=True).to_csv(correlation_path)
    outputs["feature_correlations"] = correlation_path

    labels = y.map({0: "Benign", 1: "Malignant"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=labels, order=["Benign", "Malignant"], hue=labels, legend=False, palette=["#3b82f6", "#ef4444"], ax=ax)
    ax.set(title="WDBC class distribution", xlabel="Diagnosis", ylabel="Number of records")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    class_path = output_dir / "class_distribution.png"
    fig.savefig(class_path, dpi=160)
    plt.close(fig)
    outputs["class_distribution"] = class_path

    fig, axes = plt.subplots(5, 6, figsize=(20, 15))
    for ax, feature in zip(axes.flat, X.columns):
        sns.histplot(data=X, x=feature, hue=labels, bins=22, stat="density", common_norm=False, element="step", palette=["#3b82f6", "#ef4444"], ax=ax)
        ax.set_title(feature, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ax.legend_ is not None:
            ax.legend_.remove()
    fig.suptitle("WDBC feature distributions by diagnosis", y=1.01, fontsize=16)
    fig.tight_layout()
    distribution_path = output_dir / "feature_distributions.png"
    fig.savefig(distribution_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs["feature_distributions"] = distribution_path

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(X.corr(numeric_only=True), cmap="vlag", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("WDBC feature correlation matrix")
    fig.tight_layout()
    heatmap_path = output_dir / "correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)
    outputs["correlation_heatmap"] = heatmap_path

    return outputs
