"""
Publication-quality visualization suite for comparative breast cancer analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

MODEL_COLORS = {
    "TabPFN": "#2563eb",       # Royal Blue
    "CatBoost": "#ea580c",     # Amber/Orange
    "FT-Transformer": "#16a34a", # Forest Green
    "XGBoost": "#7c3aed",      # Purple
}


def plot_roc_comparison(models_data: dict[str, dict], output_path: Path) -> None:
    """Plot multi-model ROC curves with AUC in legend."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance baseline (AUC = 0.50)")

    for name, metrics in models_data.items():
        roc = metrics.get("roc_curve", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        auc_score = metrics.get("roc_auc", 0.0)
        color = MODEL_COLORS.get(name, "#0f172a")

        if fpr and tpr:
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=color, linewidth=2.2)

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def plot_pr_comparison(models_data: dict[str, dict], output_path: Path) -> None:
    """Plot Precision-Recall curves with Average Precision in legend."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, metrics in models_data.items():
        pr = metrics.get("pr_curve", {})
        prec = pr.get("precision", [])
        rec = pr.get("recall", [])
        ap = metrics.get("avg_precision", 0.0)
        color = MODEL_COLORS.get(name, "#0f172a")

        if prec and rec:
            ax.plot(rec, prec, label=f"{name} (AP = {ap:.4f})", color=color, linewidth=2.2)

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def plot_confusion_matrices(models_data: dict[str, dict], output_path: Path) -> None:
    """Plot 2x2 grid of confusion matrices on the test partition."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (name, metrics) in zip(axes.flat, models_data.items()):
        cm = metrics.get("confusion_matrix", {})
        matrix = np.array([
            [cm.get("tn", 0), cm.get("fp", 0)],
            [cm.get("fn", 0), cm.get("tp", 0)],
        ])
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Benign", "Malignant"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(
            f"{name}\nAcc: {metrics.get('accuracy', 0):.4f} | F1: {metrics.get('f1_score', 0):.4f}",
            fontweight="bold",
            fontsize=11,
        )

    fig.suptitle("Confusion Matrix Grid — Test Set Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def plot_metric_bars(models_data: dict[str, dict], output_path: Path) -> None:
    """Grouped bar chart for primary metrics across models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_keys = ["accuracy", "precision", "f1_score", "recall", "roc_auc", "mcc"]
    metric_labels = ["Accuracy", "Precision", "F1-Score", "Recall", "ROC-AUC", "MCC"]

    model_names = list(models_data.keys())
    x = np.arange(len(metrics_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, name in enumerate(model_names):
        vals = [models_data[name].get(k, 0.0) for k in metrics_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        color = MODEL_COLORS.get(name, "#334155")
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_ylim([0.85, 1.08])
    ax.set_title("Comparative Evaluation Metrics on Held-Out Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def plot_calibration_curves(calib_results: dict[str, dict], output_path: Path) -> None:
    """Plot reliability diagram comparing before and after calibration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model_name, c_data) in zip(axes, calib_results.items()):
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        styles = [
            ("raw", "Raw Uncalibrated", "#ef4444", "-"),
            ("platt", "Platt Scaled", "#2563eb", "--"),
            ("isotonic", "Isotonic Regression", "#16a34a", "-."),
        ]
        for key, lbl, color, ls in styles:
            method_data = c_data.get(key, {})
            curve = method_data.get("curve", {})
            ece = method_data.get("ece", 0.0)
            x_vals = curve.get("mean_predicted_value", [])
            y_vals = curve.get("fraction_of_positives", [])
            if x_vals and y_vals:
                ax.plot(x_vals, y_vals, label=f"{lbl} (ECE={ece:.3f})", color=color, linestyle=ls, marker="o", linewidth=1.8)

        ax.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
        ax.set_ylabel("Fraction of True Malignants", fontsize=10, fontweight="bold")
        ax.set_title(f"Reliability Diagram: {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Probability Calibration Analysis (Pre- vs Post-Calibration)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")
