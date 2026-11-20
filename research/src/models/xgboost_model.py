"""XGBoost gradient boosting classifier with optional Optuna hyperparameter optimization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model wrapper supporting Optuna tuning and SHAP TreeExplainer."""

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    def __init__(
        self,
        params: dict | None = None,
        tune: bool = False,
        n_trials: int = 25,
        random_state: int = 42,
    ):
        super().__init__(name="XGBoost", random_state=random_state)
        self.params = {**self.DEFAULT_PARAMS, "random_state": random_state}
        if params:
            self.params.update(params)
        self.tune = tune
        self.n_trials = n_trials
        self.best_params_: dict = {}
        self._scaler = StandardScaler()

    def _build_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(**self.params)

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        try:
            import optuna
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from xgboost import XGBClassifier

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                p = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                    "eval_metric": "logloss",
                    "random_state": self.random_state,
                }
                clf = XGBClassifier(**p)
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            return study.best_params
        except Exception:
            # Fall back to default params if optuna is not yet installed or encounters an issue
            return {}

    def _fit_impl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        from xgboost import XGBClassifier

        X_tr = self._scaler.fit_transform(X_train)
        y_tr = y_train.values if hasattr(y_train, "values") else y_train

        if self.tune:
            best = self._tune_hyperparameters(X_tr, y_tr)
            if best:
                self.best_params_ = best
                self.params.update(best)
                self._model = XGBClassifier(**self.params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_v = self._scaler.transform(X_val)
            y_v = y_val.values if hasattr(y_val, "values") else y_val
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            fit_kwargs["verbose"] = False

        self._model.fit(X_tr, y_tr, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        return self._model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        return self._model.predict_proba(X_arr)

    def get_feature_importances(self, feature_names: list[str]) -> pd.Series:
        """Return native feature importances sorted descending."""
        importances = self._model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def get_sklearn_estimator(self):
        """Return the underlying booster for SHAP TreeExplainer."""
        return self._model

    def get_scaler(self):
        return self._scaler
