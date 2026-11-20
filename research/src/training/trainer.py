"""
Master training orchestrator for the comparative breast cancer ML framework.

Pipeline:
1. Load & validate WDBC dataset (569 rows, 30 features, M/B diagnosis)
2. Stratified 80/20 train/test split
3. Stratified 5-fold CV on training data for each model:
   - TabPFN-2.5 (Primary candidate model)
   - CatBoost (Gradient boosting benchmark)
   - FT-Transformer (Deep learning tabular benchmark)
   - XGBoost (Gradient boosting benchmark)
4. Model fitting on training partition & evaluation on held-out test partition
5. Calibration analysis (Platt Scaling, Isotonic Regression, ECE calculation)
6. Explainable AI (SHAP analysis & feature importance)
7. Visualization generation (ROC, PR, confusion matrix, metric comparisons)
8. Artifact serialization (models, plots, metrics JSON)
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import sys
RESEARCH_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_DIR))

import config as cfg
from src.data.loader import load_wdbc
from src.data.splitter import make_train_test_split
from src.models.tabpfn_model import TabPFNModel
from src.models.catboost_model import CatBoostModel
from src.models.ft_transformer import FTTransformerModel
from src.models.xgboost_model import XGBoostModel
from src.training.cross_validation import run_cross_validation
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.calibration import evaluate_calibration_methods
from src.evaluation.comparison import generate_comparison_table, rank_models_by_priority
from src.explainability.shap_analysis import create_shap_explainer, compute_shap_values, plot_shap_summary
from src.explainability.feature_importance import compute_global_shap_importance, plot_top10_comparison
from src.visualization.plots import (
    plot_roc_comparison,
    plot_pr_comparison,
    plot_confusion_matrices,
    plot_metric_bars,
    plot_calibration_curves,
)


def get_model_instances(random_state: int = 42, tune: bool = False, n_trials: int = 25) -> dict:
    """Return dictionary of un-fitted model instances."""
    return {
        "TabPFN": TabPFNModel(random_state=random_state),
        "CatBoost": CatBoostModel(tune=tune, n_trials=n_trials, random_state=random_state),
        "FT-Transformer": FTTransformerModel(random_state=random_state),
        "XGBoost": XGBoostModel(tune=tune, n_trials=n_trials, random_state=random_state),
    }


def run_full_research_pipeline(
    wdbc_path: Path | None = None,
    tune: bool = False,
    n_trials: int = 25,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Execute the end-to-end comparative research pipeline.
    """
    cfg.ensure_directories()
    wdbc_path = wdbc_path or cfg.WDBC_PATH

    if verbose:
        print("=" * 70)
        print("Starting Breast Cancer Research Comparative Experiment")
        print("Models: TabPFN-2.5 vs CatBoost vs FT-Transformer vs XGBoost")
        print("=" * 70)

    # 1. Load Data
    X, y, feature_names, metadata = load_wdbc(wdbc_path)
    if verbose:
        print(f"Loaded {len(X)} samples with {len(feature_names)} features.")
        print(f"Class distribution: {metadata['class_distribution']}")

    # 2. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=random_state
    )

    models_dict = get_model_instances(random_state=random_state, tune=tune, n_trials=n_trials)
    experiment_results = {}
    fitted_models = {}
    calib_results = {}
    shap_results = {}
    top_importances = {}

    for name, _ in models_dict.items():
        if verbose:
            print(f"\nEvaluating Model: {name}")
            print(f"  Running 5-fold cross-validation on training data...")

        # Cross-validation
        def model_factory(m_name=name):
            insts = get_model_instances(random_state=random_state, tune=tune, n_trials=n_trials)
            return insts[m_name]

        cv_data = run_cross_validation(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            n_splits=cfg.N_FOLDS,
            random_state=random_state,
            model_name=name,
        )

        # Fit final model on full training set
        if verbose:
            print(f"  Fitting {name} on full training set ({len(X_train)} samples)...")
        final_model = model_factory()
        t0 = time.time()
        final_model.fit(X_train, y_train)
        final_fit_time = time.time() - t0

        # Evaluate on untouched test set
        t1 = time.time()
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        final_infer_time = time.time() - t1

        test_metrics = compute_all_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=name,
        )

        # Save fitted model
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        model_save_path = cfg.MODELS_DIR / f"{safe_name}.joblib"
        final_model.save(model_save_path)
        fitted_models[name] = final_model

        # Calibration analysis (fit calibrator on cross-validated OOF probabilities, evaluate on test set)
        oof_probs = np.array(cv_data["oof_probabilities"])
        c_eval = evaluate_calibration_methods(
            y_train_calib=y_train.values,
            y_prob_train_calib=oof_probs,
            y_test=y_test.values,
            y_prob_test=y_prob,
            n_bins=10,
        )
        calib_results[name] = c_eval

        # SHAP & Feature Importance
        if verbose:
            print(f"  Computing SHAP explainability for {name}...")
        try:
            explainer, explainer_type = create_shap_explainer(final_model, name, X_train)
            shap_vals, X_eval = compute_shap_values(explainer, X_test, explainer_type)

            # Generate SHAP plots
            beeswarm_path = cfg.REPORTS_DIR / f"{safe_name}_shap_beeswarm.png"
            plot_shap_summary(shap_vals, X_eval, name, beeswarm_path, plot_type="dot")

            bar_path = cfg.REPORTS_DIR / f"{safe_name}_shap_bar.png"
            plot_shap_summary(shap_vals, X_eval, name, bar_path, plot_type="bar")

            # Global SHAP importance
            global_imp = compute_global_shap_importance(shap_vals, feature_names)
            top_importances[name] = global_imp
            shap_results[name] = {
                "explainer_type": explainer_type,
                "beeswarm_plot": str(beeswarm_path),
                "bar_plot": str(bar_path),
                "top_5_features": global_imp.head(5).to_dict(),
            }
        except Exception as e:
            if verbose:
                print(f"  SHAP computation warning for {name}: {e}")
            shap_results[name] = {"error": str(e)}

        experiment_results[name] = {
            "test_metrics": test_metrics,
            "cv_summary": cv_data["cv_summary"],
            "cv_fold_results": cv_data["fold_results"],
            "final_train_time": round(final_fit_time, 4),
            "final_infer_time": round(final_infer_time, 4),
            "model_path": str(model_save_path),
            "calibration": c_eval,
            "shap": shap_results[name],
        }

        if verbose:
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f} | Recall: {test_metrics['recall']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Generate Comparison Figures
    if verbose:
        print("\nGenerating consolidated research plots...")

    test_metrics_all = {m: experiment_results[m]["test_metrics"] for m in models_dict}
    plot_roc_comparison(test_metrics_all, cfg.REPORTS_DIR / "roc_curves_comparison.png")
    plot_pr_comparison(test_metrics_all, cfg.REPORTS_DIR / "pr_curves_comparison.png")
    plot_confusion_matrices(test_metrics_all, cfg.REPORTS_DIR / "confusion_matrices_grid.png")
    plot_metric_bars(test_metrics_all, cfg.REPORTS_DIR / "metrics_comparison_bars.png")

    if len(calib_results) >= 2:
        # Plot calibration for first two / TabPFN & benchmark
        selected_calib = {k: calib_results[k] for k in list(calib_results)[:2]}
        plot_calibration_curves(selected_calib, cfg.REPORTS_DIR / "calibration_comparison.png")

    if top_importances:
        plot_top10_comparison(top_importances, cfg.REPORTS_DIR / "feature_importance_top10.png")

    # Generate consolidated table and rankings
    comp_df = generate_comparison_table(test_metrics_all)
    rank_df = rank_models_by_priority(test_metrics_all)
    comp_df.to_csv(cfg.REPORTS_DIR / "model_comparison_table.csv", index=False)
    rank_df.to_csv(cfg.REPORTS_DIR / "model_rankings.csv", index=False)

    # Save complete JSON
    def serialize_obj(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output_payload = {
        "metadata": metadata,
        "models": experiment_results,
        "comparison_table": comp_df.to_dict(orient="records"),
        "ranking": rank_df.to_dict(orient="records"),
    }
    results_json_path = cfg.RESULTS_DIR / "experiment_results.json"
    results_json_path.write_text(json.dumps(output_payload, indent=2, default=serialize_obj), encoding="utf-8")

    if verbose:
        print(f"\nSaved experiment results to: {results_json_path}")
        print("\nConsolidated Comparison Table:")
        print(comp_df.to_string(index=False))
        print("\nPriority Order Ranking:")
        print(rank_df.to_string(index=False))

    return output_payload
