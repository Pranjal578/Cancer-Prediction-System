"""
Explainable Artificial Intelligence (XAI) module powered by SHAP.

Model-appropriate explainers:
- XGBoost & CatBoost: TreeExplainer (exact Shapley values from decision trees)
- TabPFN: Model-agnostic KernelExplainer with background k-means/subsample
- FT-Transformer: KernelExplainer with standardized background
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


def create_shap_explainer(model, model_name: str, X_background: pd.DataFrame):
    """
    Construct the optimal SHAP explainer for the given model architecture.
    """
    import shap

    if model_name in ("CatBoost", "XGBoost"):
        try:
            estimator = model.get_sklearn_estimator()
            explainer = shap.TreeExplainer(estimator)
            return explainer, "tree"
        except Exception as e:
            warnings.warn(f"TreeExplainer failed for {model_name} ({e}); falling back to KernelExplainer.")

    # Prepare background dataset (subsample to 5 representative samples for KernelExplainer efficiency)
    X_bg = X_background.values if hasattr(X_background, "values") else X_background
    if len(X_bg) > 5:
        np.random.seed(42)
        idx = np.random.choice(len(X_bg), 5, replace=False)
        X_bg = X_bg[idx]

    def predict_fn(data):
        df = pd.DataFrame(data, columns=X_background.columns if hasattr(X_background, "columns") else None)
        return model.predict_proba(df)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, X_bg)
    return explainer, "kernel"


def compute_shap_values(
    explainer,
    X: pd.DataFrame,
    explainer_type: str,
    max_evals: int = 10,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for the positive class (Malignant).
    Returns (shap_values, X_evaluated).
    """
    import shap

    if explainer_type == "tree":
        X_arr = X.values if hasattr(X, "values") else X
        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list) and len(sv) == 2:
            return np.asarray(sv[1]), X
        if isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            return sv[:, :, 1], X
        return np.asarray(sv), X
    else:
        # Kernel explainer (subsample for tractable computational runtime)
        X_eval = X.copy()
        if len(X_eval) > 5:
            X_eval = X_eval.iloc[:5].copy()
        X_arr = X_eval.values if hasattr(X_eval, "values") else X_eval

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = explainer.shap_values(X_arr, nsamples=min(max_evals, 10))
        if isinstance(sv, list):
            return np.asarray(sv[1]), X_eval
        return np.asarray(sv), X_eval


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    model_name: str,
    output_path: Path,
    plot_type: str = "dot",
) -> None:
    """
    Generate and save a SHAP summary or beeswarm plot.
    """
    import shap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    X_df = X.copy()
    if len(X_df) > len(shap_values):
        X_df = X_df.iloc[:len(shap_values)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(
            shap_values,
            X_df,
            plot_type=plot_type,
            show=False,
            max_display=15,
        )

    title = f"SHAP Beeswarm Summary — {model_name}" if plot_type == "dot" else f"SHAP Feature Importance — {model_name}"
    plt.title(title, pad=15, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")


def plot_individual_waterfall(
    shap_values_single: np.ndarray,
    feature_values_single: np.ndarray,
    feature_names: list[str],
    base_value: float,
    predicted_prob: float,
    model_name: str,
    output_path: Path,
    max_features: int = 12,
) -> None:
    """
    Generate an individual prediction waterfall chart showing positive vs negative contributions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(np.abs(shap_values_single))[-max_features:]
    features = [feature_names[i] for i in sorted_idx]
    values = shap_values_single[sorted_idx]
    raw_vals = [feature_values_single[i] for i in sorted_idx]

    labels = [f"{f} = {v:.2f}" for f, v in zip(features, raw_vals)]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in values]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    ax.axvline(0, color="#1e293b", linestyle="--", linewidth=1.2)

    ax.set_xlabel("SHAP Value (Impact on Malignant Probability)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Individual Prediction Explanation — {model_name}\n"
        f"Predicted P(Malignant) = {predicted_prob:.3f} | Base Rate = {base_value:.3f}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")
