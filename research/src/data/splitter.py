"""Stratified train/test splitting and cross-validation partition utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split.
    
    Default: 80% development/training, 20% untouched held-out testing.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def make_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified k-fold index splits."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))
