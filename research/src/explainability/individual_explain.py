"""Per-prediction (individual) explanation module.

Generates local explanations for individual predictions using SHAP values
and feature contribution analysis. Supports a 'explain one sample' workflow
suitable for the FastAPI /explain endpoint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base_model import BaseModel


def explain_single_prediction(
    model: BaseModel,
    sample: pd.DataFrame,
    X_background: pd.DataFrame,
    feature_names: list[str],
    class_names: list[str] | None = None,
    n_background: int = 100,
) -> dict:
    """Compute a local SHAP explanation for a single sample.

    Parameters
    ----------
    model:
        A fitted BaseModel instance.
    sample:
        One-row DataFrame (shape (1, 30)) representing the sample to explain.
    X_background:
        Background dataset for the SHAP explainer.
    feature_names:
        List of 30 WDBC feature names.
    class_names:
        Optional ['Benign', 'Malignant']. Used for display only.
    n_background:
        Number of background samples to subsample for KernelExplainer.

    Returns
    -------
    dict with keys:
        - 'predicted_class': int (0 or 1)
        - 'predicted_label': str ('Benign' or 'Malignant')
        - 'probabilities': dict {'Benign': float, 'Malignant': float}
        - 'shap_values': dict mapping feature_name -> shap_value (float)
        - 'top_features': list of (feature_name, shap_value) sorted by |shap|
        - 'base_value': float (expected model output / SHAP base value)

    Raises
    ------
    NotImplementedError
        Until this stub is implemented.
    """
    raise NotImplementedError("TODO")


def format_explanation_for_api(
    explanation: dict,
) -> dict:
    """Reformat an explanation dict for JSON serialization in the FastAPI response.

    Ensures all numpy scalars are converted to Python native types.

    Parameters
    ----------
    explanation:
        Output of explain_single_prediction.

    Returns
    -------
    JSON-serializable dict.

    Raises
    ------
    NotImplementedError
        Until this stub is implemented.
    """
    raise NotImplementedError("TODO")
