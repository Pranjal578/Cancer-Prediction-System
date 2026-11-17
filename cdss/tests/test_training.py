"""Tests for the five-model comparison and model-selection workflow."""

from __future__ import annotations

import json

import joblib

from src.pipeline.data_loader import load_breast_dataset
from src.training.trainer import build_model_pipelines, train_and_compare_models


EXPECTED_MODELS = {
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
}


def test_all_five_proposed_classifiers_are_available():
    """Model construction is limited to the algorithms named in the proposal."""
    assert set(build_model_pipelines()) == EXPECTED_MODELS


def test_training_compares_selects_and_serializes_a_model(tmp_path):
    """Each required model is evaluated and a selected pipeline is saved."""
    X, y, feature_names, metadata = load_breast_dataset()
    outputs = train_and_compare_models(X, y, feature_names, tmp_path, metadata)

    comparison = outputs["comparison"]
    assert set(comparison["model"]) == EXPECTED_MODELS
    assert list(comparison.columns) == ["model", "accuracy", "precision", "recall", "f1_score"]
    assert outputs["selected_model"] in EXPECTED_MODELS
    assert outputs["model_path"].exists()
    assert outputs["metadata_path"].exists()
    assert outputs["comparison_chart"].exists()
    assert outputs["confusion_chart"].exists()

    selected_pipeline = joblib.load(outputs["model_path"])
    assert int(selected_pipeline.predict(X.head(1))[0]) in {0, 1}

    saved_metadata = json.loads(outputs["metadata_path"].read_text(encoding="utf-8"))
    assert saved_metadata["selected_model"] == outputs["selected_model"]
    assert set(saved_metadata["confusion_matrices"]) == EXPECTED_MODELS
