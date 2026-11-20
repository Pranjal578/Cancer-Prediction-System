"""Tests for evaluation metrics and probability calibration."""
import numpy as np
import pytest
from pathlib import Path
import sys

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from src.evaluation.metrics import (
    compute_ece,
    compute_specificity,
    compute_npv,
    compute_all_metrics,
)
from src.evaluation.calibration import apply_platt_scaling, apply_isotonic_regression


def test_ece_perfect_calibration():
    # If predicted probability matches actual proportion, ECE should be near 0
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.95, 0.95])
    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 0.10


def test_compute_all_metrics_keys():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.85])

    metrics = compute_all_metrics(y_true, y_pred, y_prob, model_name="TestModel")

    required_keys = [
        "accuracy", "precision", "f1_score", "recall", "roc_auc",
        "sensitivity", "specificity", "balanced_accuracy", "mcc",
        "npv", "log_loss", "brier_score", "ece", "confusion_matrix"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing evaluation metric: {key}"
        if key != "confusion_matrix":
            assert isinstance(metrics[key], (float, int))


def test_calibration_scaling():
    y_calib = np.array([0, 0, 0, 1, 1, 1])
    p_calib = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    p_eval = np.array([0.25, 0.75])

    platt_out = apply_platt_scaling(y_calib, p_calib, p_eval)
    assert len(platt_out) == 2
    assert (platt_out >= 0).all() and (platt_out <= 1).all()

    iso_out = apply_isotonic_regression(y_calib, p_calib, p_eval)
    assert len(iso_out) == 2
    assert (iso_out >= 0).all() and (iso_out <= 1).all()
