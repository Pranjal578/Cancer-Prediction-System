"""Run the breast-cancer preprocessing, EDA, model comparison, and selection flow.

Run from ``cdss/``:
    python train_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

from config import ARTIFACTS_DIR, REPORTS_DIR, ensure_project_directories
from src.analysis.eda import create_eda_outputs
from src.pipeline.data_loader import load_breast_dataset
from src.training.trainer import train_and_compare_models


def run_training(output_dir: Path | None = None) -> dict:
    """Build all artefacts for the proposed breast-cancer prediction system."""
    ensure_project_directories()
    artifacts_dir = output_dir or ARTIFACTS_DIR
    reports_dir = artifacts_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, dataset_metadata = load_breast_dataset()
    eda_outputs = create_eda_outputs(X, y, reports_dir)
    training_outputs = train_and_compare_models(
        X=X,
        y=y,
        feature_names=feature_names,
        output_dir=artifacts_dir,
        dataset_metadata=dataset_metadata,
    )

    run_summary = {
        "dataset_metadata": dataset_metadata,
        "selected_model": training_outputs["selected_model"],
        "eda_outputs": {name: str(path) for name, path in eda_outputs.items()},
        "model_comparison": str(training_outputs["comparison_csv"]),
        "best_model": str(training_outputs["model_path"]),
    }
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    return {**run_summary, **training_outputs}


def main() -> None:
    outputs = run_training()
    print("\nBreast cancer training complete")
    print(f"Selected model: {outputs['selected_model']}")
    print("\nModel comparison:")
    print(outputs["comparison"].to_string(index=False))
    print(f"\nSaved model: {outputs['model_path']}")
    print(f"Saved reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
