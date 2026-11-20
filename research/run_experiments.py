"""
Master experiment runner script for Breast Cancer Prediction & XAI Framework.

Executes:
1. WDBC data validation
2. 5-fold CV for all four models (TabPFN, CatBoost, FT-Transformer, XGBoost)
3. Model evaluation on untouched test set across 12 metrics + ECE
4. Probability calibration analysis
5. SHAP analysis and feature importance
6. Consolidated report and visualization generation

Usage:
  cd research
  ../cdss/.venv/bin/python run_experiments.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add research module to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.trainer import run_full_research_pipeline
import config as cfg


def main():
    print("=" * 70)
    print("A Comparative Machine Learning and Explainable AI Framework")
    print("Breast Cancer Classification: TabPFN-2.5 vs CatBoost vs FT-Transformer vs XGBoost")
    print("=" * 70)

    results = run_full_research_pipeline(
        wdbc_path=cfg.WDBC_PATH,
        tune=False,        # Use default benchmark hyperparameters (set True for Optuna tuning)
        n_trials=25,
        random_state=cfg.RANDOM_STATE,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Results stored at: {cfg.RESULTS_DIR / 'experiment_results.json'}")
    print(f"Visualizations stored at: {cfg.REPORTS_DIR}")
    print("\nTo launch the interactive research workbench:")
    print("  streamlit run app.py --server.port 8501")


if __name__ == "__main__":
    main()
