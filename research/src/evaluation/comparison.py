"""
Consolidated comparison table and automated model ranking.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "f1_score": "F1-Score",
    "recall": "Recall",
    "roc_auc": "ROC-AUC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced Accuracy",
    "mcc": "MCC",
    "npv": "NPV",
    "brier_score": "Brier Score",
    "log_loss": "Log Loss",
    "ece": "ECE",
}

# Lower is better for error metrics
LOWER_IS_BETTER = {"brier_score", "log_loss", "ece"}


def generate_comparison_table(metrics_per_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Construct the consolidated comparison DataFrame.
    
    Includes all 4 models as columns (TabPFN, CatBoost, FT-Transformer, XGBoost),
    all 12 metrics as rows, and an automatic 'Best Model' indicator column.
    """
    model_names = list(metrics_per_model.keys())
    rows = []

    for metric_key, metric_display in METRIC_LABELS.items():
        row_dict = {"Metric": metric_display}
        scores = {}

        for m_name in model_names:
            val = metrics_per_model[m_name].get(metric_key, None)
            row_dict[m_name] = val
            if val is not None:
                scores[m_name] = val

        if scores:
            if metric_key in LOWER_IS_BETTER:
                best_model = min(scores, key=scores.get)
            else:
                best_model = max(scores, key=scores.get)
            row_dict["Best Model"] = best_model
        else:
            row_dict["Best Model"] = "N/A"

        rows.append(row_dict)

    return pd.DataFrame(rows)


def rank_models_by_priority(metrics_per_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Rank models according to the specification priority order:
    1. Accuracy
    2. Precision
    3. F1-Score
    4. Recall
    5. ROC-AUC
    """
    priority_metrics = ["accuracy", "precision", "f1_score", "recall", "roc_auc"]
    records = []

    for m_name, m_dict in metrics_per_model.items():
        rec = {"Model": m_name}
        for pm in priority_metrics:
            rec[METRIC_LABELS[pm]] = m_dict.get(pm, 0.0)
        records.append(rec)

    df = pd.DataFrame(records)
    sort_cols = [METRIC_LABELS[pm] for pm in priority_metrics]
    return df.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
