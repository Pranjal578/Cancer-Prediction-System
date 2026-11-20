"""
Stratified 5-fold cross-validation engine for research models.

Ensures strict zero-leakage:
- Feature scalers and preprocessors are fit inside each fold on training samples only.
- Validation samples are strictly evaluated on fold-fitted models.
- Per-fold and aggregated (mean +/- std) metrics are recorded.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Callable, Any
from sklearn.model_selection import StratifiedKFold

from ..evaluation.metrics import compute_all_metrics


def run_cross_validation(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    model_name: str = "",
) -> dict[str, Any]:
    """
    Execute stratified 5-fold CV for a model.
    
    Args:
        model_factory: Callable returning a fresh, un-fitted BaseModel instance.
        X: Training features DataFrame
        y: Training labels Series
        n_splits: Number of folds (default 5)
        random_state: Reproducibility seed
        model_name: Identifier for logs and reporting
    
    Returns:
        dict containing per-fold metric dictionaries and aggregated mean/std statistics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    oof_predictions = np.zeros(len(y))
    oof_probabilities = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y.iloc[train_idx].reset_index(drop=True)
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y.iloc[val_idx].reset_index(drop=True)

        model = model_factory()
        t0 = time.time()
        model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        fit_time = time.time() - t0

        t1 = time.time()
        preds = model.predict(X_val_fold)
        probs = model.predict_proba(X_val_fold)[:, 1]
        eval_time = time.time() - t1

        oof_predictions[val_idx] = preds
        oof_probabilities[val_idx] = probs

        fold_metrics = compute_all_metrics(
            y_true=y_val_fold.values,
            y_pred=preds,
            y_prob=probs,
            model_name=f"{model_name}_fold_{fold_idx}",
        )
        fold_metrics["fold"] = fold_idx + 1
        fold_metrics["train_time"] = round(fit_time, 4)
        fold_metrics["infer_time"] = round(eval_time, 4)
        fold_results.append(fold_metrics)

    # Compute overall Out-Of-Fold (OOF) metrics
    oof_metrics = compute_all_metrics(
        y_true=y.values,
        y_pred=oof_predictions.astype(int),
        y_prob=oof_probabilities,
        model_name=f"{model_name}_oof",
    )

    # Aggregate metric means and standard deviations across folds
    tracked_metrics = [
        "accuracy", "precision", "f1_score", "recall", "roc_auc",
        "sensitivity", "specificity", "balanced_accuracy", "mcc",
        "npv", "log_loss", "brier_score", "ece",
    ]
    cv_summary = {}
    for m in tracked_metrics:
        vals = [f[m] for f in fold_results if m in f]
        cv_summary[m] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "values": [round(float(v), 4) for v in vals],
        }

    cv_summary["train_time"] = {
        "mean": round(float(np.mean([f["train_time"] for f in fold_results])), 4),
        "std": round(float(np.std([f["train_time"] for f in fold_results])), 4),
    }

    return {
        "model": model_name,
        "fold_results": fold_results,
        "cv_summary": cv_summary,
        "oof_metrics": oof_metrics,
        "oof_predictions": oof_predictions.tolist(),
        "oof_probabilities": oof_probabilities.tolist(),
    }
