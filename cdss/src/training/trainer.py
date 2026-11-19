"""Train, evaluate, compare, and save the five proposed classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from src.pipeline.preprocessor import build_preprocessor

SELECTION_RULE = (
    "Highest validation recall; ties are resolved by F1-score, precision, then accuracy."
)


def build_model_pipelines() -> dict[str, Pipeline]:
    """Build the five classifiers required by the proposed system.

    Each classifier receives its own imputation-and-scaling pipeline to prevent
    test-set information leaking into training.
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf"), method="sigmoid", cv=5
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }
    return {
        name: Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", classifier)]
        )
        for name, classifier in classifiers.items()
    }


def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the comparison metrics specified in the proposal."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _save_comparison_chart(results: pd.DataFrame, output_path: Path) -> None:
    """Save a grouped bar chart for the validation comparison metrics."""
    chart_data = results.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    ax = chart_data.plot(kind="bar", figsize=(12, 6), color=["#2563eb", "#7c3aed", "#ea580c", "#16a34a"])
    ax.set_title("Model comparison on the validation set")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _save_confusion_matrices(
    confusion_matrices: dict[str, list[list[int]]], output_path: Path
) -> None:
    """Save one validation confusion-matrix panel for each classifier."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (name, matrix) in zip(axes.flat, confusion_matrices.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix), display_labels=["Benign", "Malignant"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    axes.flat[-1].axis("off")
    fig.suptitle("Confusion matrices — validation set", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    output_dir: Path,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare on validation data, select by recall, then test only the winner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    comparison_rows: list[dict[str, Any]] = []
    matrices: dict[str, list[list[int]]] = {}
    for model_name, model in build_model_pipelines().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        comparison_rows.append(
            {"model": model_name, **_calculate_metrics(y_validation, predictions)}
        )
        matrices[model_name] = confusion_matrix(
            y_validation, predictions, labels=[0, 1]
        ).tolist()

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["recall", "f1_score", "precision", "accuracy"], ascending=False
    ).reset_index(drop=True)
    selected_model_name = str(comparison.iloc[0]["model"])
    deployment_models = build_model_pipelines()
    for model in deployment_models.values():
        model.fit(X_development, y_development)
    selected_model = deployment_models[selected_model_name]
    final_test_metrics = _calculate_metrics(y_test, selected_model.predict(X_test))

    comparison_csv = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    comparison_chart = output_dir / "model_comparison.png"
    _save_comparison_chart(comparison, comparison_chart)
    confusion_chart = output_dir / "confusion_matrices.png"
    _save_confusion_matrices(matrices, confusion_chart)

    model_path = output_dir / "best_model.joblib"
    joblib.dump(selected_model, model_path)
    model_pipelines_path = output_dir / "model_pipelines.joblib"
    joblib.dump(deployment_models, model_pipelines_path)
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "feature_names": feature_names,
        "training_set_size": int(len(X_train)),
        "validation_set_size": int(len(X_validation)),
        "test_set_size": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "selection_metric": "recall",
        "selection_rule": SELECTION_RULE,
        "selected_model": selected_model_name,
        "selected_model_validation_metrics": comparison.iloc[0].drop(labels="model").to_dict(),
        "selected_model_test_metrics": final_test_metrics,
        "all_model_validation_metrics": comparison.to_dict(orient="records"),
        "validation_confusion_matrices": matrices,
        "available_prediction_models": list(deployment_models),
        "dataset_metadata": dataset_metadata or {},
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "selected_model": selected_model_name,
        "comparison": comparison,
        "validation_confusion_matrices": matrices,
        "selected_model_test_metrics": final_test_metrics,
        "model_path": model_path,
        "model_pipelines_path": model_pipelines_path,
        "metadata_path": metadata_path,
        "comparison_csv": comparison_csv,
        "comparison_chart": comparison_chart,
        "confusion_chart": confusion_chart,
    }
