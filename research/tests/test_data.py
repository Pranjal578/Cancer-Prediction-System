"""Tests for WDBC dataset loading, validation, and splitting."""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

import sys
TESTS_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = TESTS_DIR.parent
sys.path.insert(0, str(RESEARCH_DIR))

import config as cfg
from src.data.loader import load_wdbc, WDBC_FEATURE_NAMES
from src.data.splitter import make_train_test_split, make_cv_folds


def test_wdbc_file_exists():
    assert cfg.WDBC_PATH.exists(), f"WDBC raw file missing at {cfg.WDBC_PATH}"


def test_wdbc_loading_structure():
    X, y, feature_names, metadata = load_wdbc(cfg.WDBC_PATH)

    assert len(X) == 569, f"Expected 569 observations, got {len(X)}"
    assert len(feature_names) == 30, f"Expected 30 features, got {len(feature_names)}"
    assert X.shape == (569, 30), f"Invalid feature matrix shape: {X.shape}"
    assert len(y) == 569

    # Class balance check
    assert (y == 0).sum() == 357, "Expected 357 Benign samples"
    assert (y == 1).sum() == 212, "Expected 212 Malignant samples"

    # Numeric and non-negative
    assert np.issubdtype(X.values.dtype, np.number)
    assert not (X < 0).any().any(), "FNA features cannot be negative"


def test_train_test_splitting():
    X, y, _, _ = load_wdbc(cfg.WDBC_PATH)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, test_size=0.20, random_state=42)

    assert len(X_train) == 455
    assert len(X_test) == 114
    # Stratified balance check
    train_mal_ratio = (y_train == 1).mean()
    test_mal_ratio = (y_test == 1).mean()
    assert abs(train_mal_ratio - test_mal_ratio) < 0.02


def test_5fold_cv_splits():
    X, y, _, _ = load_wdbc(cfg.WDBC_PATH)
    folds = make_cv_folds(X, y, n_splits=5, random_state=42)

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(train_idx) + len(val_idx) == 569
        # No overlap between train and val
        assert len(set(train_idx).intersection(set(val_idx))) == 0
