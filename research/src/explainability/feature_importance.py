"""
Multi-angle feature importance analysis:
1. Native model feature importance (CatBoost, XGBoost)
2. Permutation feature importance (model-agnostic)
3. SHAP-based global feature importance (mean absolute SHAP)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """
    Calculate permutation feature importance using drop in ROC-AUC.
    """
    from sklearn.metrics import roc_auc_score

    baseline_prob = model.predict_proba(X_val)[:, 1]
    baseline_auc = roc_auc_score(y_val, baseline_prob)

    rng = np.random.RandomState(random_state)
    importances = []

    for col in feature_names:
        scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            shuffled = X_perm[col].values.copy()
            rng.shuffle(shuffled)
            X_perm[col] = shuffled
            perm_prob = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y_val, perm_prob)
            scores.append(baseline_auc - perm_auc)
        importances.append(float(np.mean(scores)))

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def compute_global_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.Series:
    """
    Calculate mean absolute SHAP value per feature:
    I_j = (1/N) * sum_i |phi_{i,j}|
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)


def plot_top10_comparison(
    importances_by_model: dict[str, pd.Series],
    output_path: Path,
    title: str = "Top 10 Most Influential Tumour Features Across Models",
) -> None:
    """
    Plot top 10 features for all models side-by-side.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_models = len(importances_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 6), sharey=False)
    if n_models == 1:
        axes = [axes]

    colors = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed"]

    for ax, (name, s), color in zip(axes, importances_by_model.items(), colors):
        top10 = s.head(10).iloc[::-1]
        ax.barh(top10.index, top10.values, color=color, alpha=0.85)
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close("all")
