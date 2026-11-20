"""Load and validate the Breast Cancer Wisconsin (Diagnostic) dataset.

Column layout (from UCI documentation):
  col 0: Sample ID (removed)
  col 1: Diagnosis — M (Malignant) or B (Benign)
  col 2–31: 30 real-valued features (mean, se, worst × 10 FNA cell measurements)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

WDBC_FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

WDBC_COLUMN_NAMES = ["id", "diagnosis"] + WDBC_FEATURE_NAMES


def load_wdbc(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """
    Load WDBC, validate, and return (X, y, feature_names, metadata).
    
    - X: DataFrame of 30 numeric features, shape (569, 30)
    - y: Series of integer labels, 0=Benign, 1=Malignant (positive class)
    - feature_names: list of 30 feature name strings
    - metadata: dictionary with dataset statistics and provenance
    """
    if not path.exists():
        raise FileNotFoundError(f"WDBC dataset not found at: {path}")

    # Read raw data without header
    df = pd.read_csv(path, header=None, names=WDBC_COLUMN_NAMES)

    # Remove non-predictive sample ID
    df = df.drop(columns=["id"])

    # Validate target column values
    valid_diag = {"M", "B"}
    unexpected = set(df["diagnosis"].unique()) - valid_diag
    if unexpected:
        raise ValueError(f"Unexpected diagnosis values encountered: {unexpected}")

    # Binary encoding: Malignant (M) = 1 (positive), Benign (B) = 0
    y = (df["diagnosis"] == "M").astype(int)
    y.name = "target"

    # Extract feature matrix
    feature_names = list(WDBC_FEATURE_NAMES)
    X = df[feature_names].copy()

    # Verify numeric conversion
    for col in feature_names:
        X[col] = pd.to_numeric(X[col], errors="raise")

    # FNA measurements cannot be strictly negative
    if (X < 0).any().any():
        raise ValueError("Negative feature values found in FNA measurements; invalid data.")

    # Check and remove exact duplicates
    n_before = len(X)
    duplicate_mask = df[feature_names + ["diagnosis"]].duplicated()
    if duplicate_mask.any():
        X = X[~duplicate_mask].reset_index(drop=True)
        y = y[~duplicate_mask].reset_index(drop=True)
    n_removed = n_before - len(X)

    metadata = {
        "source": "UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic) Dataset",
        "sample_count": len(X),
        "feature_count": len(feature_names),
        "duplicates_removed": n_removed,
        "class_distribution": {
            "Benign (0)": int((y == 0).sum()),
            "Malignant (1)": int((y == 1).sum()),
        },
        "benign_count": int((y == 0).sum()),
        "malignant_count": int((y == 1).sum()),
        "positive_class": "Malignant (1)",
        "negative_class": "Benign (0)",
        "feature_names": feature_names,
        "features_summary": {
            col: {
                "mean": round(float(X[col].mean()), 4),
                "std": round(float(X[col].std()), 4),
                "min": round(float(X[col].min()), 4),
                "median": round(float(X[col].median()), 4),
                "max": round(float(X[col].max()), 4),
            }
            for col in feature_names
        },
    }

    return X, y, feature_names, metadata
