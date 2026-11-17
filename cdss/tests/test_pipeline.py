"""Tests for WDBC loading and preprocessing."""

from __future__ import annotations

import numpy as np

from src.pipeline.data_loader import WDBC_FEATURE_COLUMNS, load_breast_dataset, load_dataset
from src.pipeline.preprocessor import build_preprocessor, has_missing_values


def test_wdbc_loader_matches_proposed_dataset_shape():
    """The application uses the requested 569-row, 30-feature WDBC data."""
    X, y, feature_names, metadata = load_breast_dataset()

    assert X.shape == (569, 30)
    assert len(y) == 569
    assert feature_names == WDBC_FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert metadata["records_after_duplicate_removal"] == 569
    assert metadata["feature_count"] == 30
    assert metadata["duplicate_rows_removed"] == 0


def test_only_breast_dataset_is_supported():
    """Strict scope excludes the previous cervical modality."""
    try:
        load_dataset("cervical")
    except ValueError as error:
        assert "Only the breast" in str(error)
    else:
        raise AssertionError("A cervical dataset should not be supported")


def test_preprocessor_imputes_and_scales_numeric_features():
    """Missing numeric values are imputed and transformed data contain no NaNs."""
    X, _, _, _ = load_breast_dataset()
    sample = X.head(8).copy()
    sample.iloc[0, 0] = np.nan

    transformed = build_preprocessor().fit_transform(sample)

    assert transformed.shape == sample.shape
    assert not has_missing_values(transformed)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
