"""
Comprehensive evaluation metrics for binary breast cancer classification.

Positive class: Malignant = 1 (tumour confirmed malignant)
Negative class: Benign = 0

Metrics implemented:
  1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
  2. Precision = TP / (TP + FP)
  3. F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
  4. Recall / Sensitivity = TP / (TP + FN)
  5. ROC-AUC = Area under Receiver Operating Characteristic curve
  6. Specificity = TN / (TN + FP)
  7. Balanced Accuracy = (Sensitivity + Specificity) / 2
  8. Matthews Correlation Coefficient (MCC)
  9. Negative Predictive Value (NPV) = TN / (TN + FN)
  10. Log Loss = Cross-entropy loss
  11. Brier Score = Mean squared difference between predicted probability and actual class
  12. Expected Calibration Error (ECE) = Weighted average calibration error across 10 equal-width bins
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width probability binning.
    
    Methodology:
    - Partitions predicted probabilities into `n_bins` equal-width intervals [0, 1/B), [1/B, 2/B), ..., [(B-1)/B, 1].
    - In each non-empty bin B_m:
        acc(B_m) = fraction of true positive samples in B_m
        conf(B_m) = average predicted positive probability in B_m
    - ECE = sum_{m=1}^{n_bins} (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    if n == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity (True Negative Rate) = TN / (TN + FP)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative Predictive Value = TN / (TN + FN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
) -> dict[str, Any]:
    """
    Compute the complete 12 research metrics, confusion matrix, and curves.
    
    Args:
        y_true: Ground truth binary labels (0=Benign, 1=Malignant)
        y_pred: Predicted class labels (0 or 1)
        y_prob: Predicted probability of positive class (Malignant)
        model_name: Model identifier string
    
    Returns:
        Structured dict with all metrics formatted to 4 decimal places
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # ROC & PR curves
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_true, y_prob)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = compute_specificity(y_true, y_pred)
    npv = compute_npv(y_true, y_pred)

    # Bounded probability for log_loss
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    return {
        "model": model_name,
        # Priority order primary evaluation metrics
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        # Additional research evaluation metrics
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "npv": round(npv, 4),
        "log_loss": round(float(log_loss(y_true, y_prob_clipped)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece": round(compute_ece(y_true, y_prob, n_bins=10), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        # Confusion matrix counts
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        # Curve arrays for plotting
        "roc_curve": {
            "fpr": [round(float(x), 5) for x in fpr],
            "tpr": [round(float(x), 5) for x in tpr],
        },
        "pr_curve": {
            "precision": [round(float(x), 5) for x in prec_curve],
            "recall": [round(float(x), 5) for x in rec_curve],
        },
    }
