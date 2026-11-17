"""Load and validate the Wisconsin Diagnostic Breast Cancer (WDBC) dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import WDBC_PATH

WDBC_COLUMNS = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]
WDBC_FEATURE_COLUMNS = WDBC_COLUMNS[2:]


def load_breast_dataset(
    path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Return WDBC features and labels ready for preprocessing.

    The loader removes the non-predictive record identifier, verifies duplicate
    rows, converts malformed numeric entries to missing values for later median
    imputation, and encodes ``M`` as 1 and ``B`` as 0.
    """
    dataset_path = Path(path) if path is not None else WDBC_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"WDBC dataset not found: {dataset_path}")

    raw = pd.read_csv(dataset_path, header=None, names=WDBC_COLUMNS)
    rows_before_deduplication = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    if duplicate_rows:
        raw = raw.drop_duplicates().reset_index(drop=True)

    diagnosis = raw["diagnosis"].astype(str).str.strip().str.upper()
    unknown_labels = sorted(set(diagnosis) - {"M", "B"})
    if unknown_labels:
        raise ValueError(f"Unexpected diagnosis labels in WDBC data: {unknown_labels}")

    features = raw[WDBC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_values = int(features.isna().sum().sum())
    target = (diagnosis == "M").astype(int)
    target.name = "malignant"

    metadata: dict[str, Any] = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository",
        "records_before_duplicate_removal": rows_before_deduplication,
        "records_after_duplicate_removal": len(features),
        "duplicate_rows_removed": duplicate_rows,
        "feature_count": len(WDBC_FEATURE_COLUMNS),
        "missing_values": missing_values,
        "target_mapping": {"Benign (B)": 0, "Malignant (M)": 1},
        "class_distribution": {
            "Benign": int((target == 0).sum()),
            "Malignant": int((target == 1).sum()),
        },
    }
    return features, target, WDBC_FEATURE_COLUMNS.copy(), metadata


def load_dataset(modality: str = "breast") -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Compatibility entry point for the single supported modality: breast."""
    if modality.lower() != "breast":
        raise ValueError("Only the breast WDBC dataset is supported by this application.")
    return load_breast_dataset()
