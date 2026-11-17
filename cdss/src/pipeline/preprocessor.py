"""Preprocessing used consistently during training and prediction."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """Return the numeric preprocessing pipeline: median imputation then scaling."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_pipeline() -> Pipeline:
    """Backward-compatible alias for the project's preprocessing pipeline."""
    return build_preprocessor()


def has_missing_values(values: np.ndarray) -> bool:
    """Return whether a transformed matrix still contains missing values."""
    return bool(np.isnan(values).any())
