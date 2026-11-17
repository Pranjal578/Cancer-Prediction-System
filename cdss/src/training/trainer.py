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

from config import RANDOM_STATE, TEST_SIZE
from src.pipeline.preprocessor import build_preprocessor

SELECTION_RULE = (
    "Highest test F1-score; ties are resolved by recall, precision, then accuracy."
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
    """Save a grouped bar chart for the four required evaluation metrics."""
    chart_data = results.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    ax = chart_data.plot(kind="bar", figsize=(12, 6), color=["#2563eb", "#7c3aed", "#ea580c", "#16a34a"])
    ax.set_title("Model comparison on the held-out test set")
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
    """Save one confusion-matrix panel for each classifier."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (name, matrix) in zip(axes.flat, confusion_matrices.items()):
        ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix), display_labels=["Benign", "Malignant"]
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    axes.flat[-1].axis("off")
    fig.suptitle("Confusion matrices — held-out test set", fontsize=16)
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
    """Train all five models, select the best one, and save reproducible outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    fitted_models: dict[str, Pipeline] = {}
    comparison_rows: list[dict[str, Any]] = []
    matrices: dict[str, list[list[int]]] = {}
    for model_name, model in build_model_pipelines().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        fitted_models[model_name] = model
        comparison_rows.append({"model": model_name, **_calculate_metrics(y_test, predictions)})
        matrices[model_name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["f1_score", "recall", "precision", "accuracy"], ascending=False
    ).reset_index(drop=True)
    selected_model_name = str(comparison.iloc[0]["model"])
    selected_model = fitted_models[selected_model_name]

    comparison_csv = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_csv, index=False)
    comparison_chart = output_dir / "model_comparison.png"
    _save_comparison_chart(comparison, comparison_chart)
    confusion_chart = output_dir / "confusion_matrices.png"
    _save_confusion_matrices(matrices, confusion_chart)

    model_path = output_dir / "best_model.joblib"
    joblib.dump(selected_model, model_path)
    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "feature_names": feature_names,
        "test_set_size": int(len(X_test)),
        "training_set_size": int(len(X_train)),
        "random_state": RANDOM_STATE,
        "selection_metric": "f1_score",
        "selection_rule": SELECTION_RULE,
        "selected_model": selected_model_name,
        "selected_model_metrics": comparison.iloc[0].drop(labels="model").to_dict(),
        "all_model_metrics": comparison.to_dict(orient="records"),
        "confusion_matrices": matrices,
        "dataset_metadata": dataset_metadata or {},
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "selected_model": selected_model_name,
        "comparison": comparison,
        "confusion_matrices": matrices,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "comparison_csv": comparison_csv,
        "comparison_chart": comparison_chart,
        "confusion_chart": confusion_chart,
    }
