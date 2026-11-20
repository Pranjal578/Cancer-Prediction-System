"""
Probability calibration analysis and evaluation.

Implements:
- Reliability diagrams (calibration curves)
- Expected Calibration Error (ECE)
- Platt Scaling (logistic calibration)
- Isotonic Regression calibration
- Pre- vs post-calibration comparison without test leakage
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import compute_ece


def apply_platt_scaling(
    y_calib: np.ndarray,
    y_prob_calib: np.ndarray,
    y_prob_eval: np.ndarray,
) -> np.ndarray:
    """
    Apply Platt Scaling: fit a univariate logistic regression on calibration probabilities.
    
    Prevents leakage by fitting only on the dedicated calibration partition.
    """
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    # Fit on log-odds or raw probability
    lr.fit(y_prob_calib.reshape(-1, 1), y_calib)
    return lr.predict_proba(y_prob_eval.reshape(-1, 1))[:, 1]


def apply_isotonic_regression(
    y_calib: np.ndarray,
    y_prob_calib: np.ndarray,
    y_prob_eval: np.ndarray,
) -> np.ndarray:
    """
    Apply Isotonic Regression: fit a non-parametric monotonic step function.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_calib, y_calib)
    return iso.predict(y_prob_eval)


def evaluate_calibration_methods(
    y_train_calib: np.ndarray,
    y_prob_train_calib: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compare raw, Platt-scaled, and Isotonic-calibrated predictions on the untouched test set.
    
    Calibration fit: performed on y_train_calib / y_prob_train_calib.
    Evaluation: performed on y_test / y_prob_test.
    """
    # Raw uncalibrated
    raw_ece = compute_ece(y_test, y_prob_test, n_bins=n_bins)
    raw_brier = float(brier_score_loss(y_test, y_prob_test))
    raw_logloss = float(log_loss(y_test, np.clip(y_prob_test, 1e-15, 1 - 1e-15)))
    raw_frac, raw_mean = calibration_curve(y_test, y_prob_test, n_bins=n_bins, strategy="uniform")

    # Platt Scaling
    platt_prob = apply_platt_scaling(y_train_calib, y_prob_train_calib, y_prob_test)
    platt_ece = compute_ece(y_test, platt_prob, n_bins=n_bins)
    platt_brier = float(brier_score_loss(y_test, platt_prob))
    platt_logloss = float(log_loss(y_test, np.clip(platt_prob, 1e-15, 1 - 1e-15)))
    platt_frac, platt_mean = calibration_curve(y_test, platt_prob, n_bins=n_bins, strategy="uniform")

    # Isotonic Regression
    iso_prob = apply_isotonic_regression(y_train_calib, y_prob_train_calib, y_prob_test)
    iso_ece = compute_ece(y_test, iso_prob, n_bins=n_bins)
    iso_brier = float(brier_score_loss(y_test, iso_prob))
    iso_logloss = float(log_loss(y_test, np.clip(iso_prob, 1e-15, 1 - 1e-15)))
    iso_frac, iso_mean = calibration_curve(y_test, iso_prob, n_bins=n_bins, strategy="uniform")

    return {
        "raw": {
            "ece": round(raw_ece, 4),
            "brier_score": round(raw_brier, 4),
            "log_loss": round(raw_logloss, 4),
            "curve": {
                "mean_predicted_value": [round(float(v), 4) for v in raw_mean],
                "fraction_of_positives": [round(float(v), 4) for v in raw_frac],
            },
        },
        "platt": {
            "ece": round(platt_ece, 4),
            "brier_score": round(platt_brier, 4),
            "log_loss": round(platt_logloss, 4),
            "curve": {
                "mean_predicted_value": [round(float(v), 4) for v in platt_mean],
                "fraction_of_positives": [round(float(v), 4) for v in platt_frac],
            },
        },
        "isotonic": {
            "ece": round(iso_ece, 4),
            "brier_score": round(iso_brier, 4),
            "log_loss": round(iso_logloss, 4),
            "curve": {
                "mean_predicted_value": [round(float(v), 4) for v in iso_mean],
                "fraction_of_positives": [round(float(v), 4) for v in iso_frac],
            },
        },
    }
