"""CatBoost gradient boosting classifier with Optuna hyperparameter optimization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost classifier wrapper with Optuna tuning and SHAP support."""

    DEFAULT_PARAMS = {
        "iterations": 400,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    }

    def __init__(
        self,
        params: dict | None = None,
        tune: bool = False,
        n_trials: int = 25,
        random_state: int = 42,
    ):
        super().__init__(name="CatBoost", random_state=random_state)
        self.params = {**self.DEFAULT_PARAMS, "random_seed": random_state}
        if params:
            self.params.update(params)
        self.tune = tune
        self.n_trials = n_trials
        self.best_params_: dict = {}
        self._scaler = StandardScaler()

    def _build_model(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self.params)

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        try:
            import optuna
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from catboost import CatBoostClassifier

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                p = {
                    "iterations": trial.suggest_int("iterations", 150, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "depth": trial.suggest_int("depth", 3, 8),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "loss_function": "Logloss",
                    "random_seed": self.random_state,
                    "verbose": 0,
                    "allow_writing_files": False,
                }
                clf = CatBoostClassifier(**p)
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
            return {}

    def _fit_impl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        from catboost import CatBoostClassifier, Pool

        X_tr = self._scaler.fit_transform(X_train)
        y_tr = y_train.values if hasattr(y_train, "values") else y_train

        if self.tune:
            best = self._tune_hyperparameters(X_tr, y_tr)
            if best:
                self.best_params_ = best
                self.params.update(best)
                self._model = CatBoostClassifier(**self.params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_v = self._scaler.transform(X_val)
            y_v = y_val.values if hasattr(y_val, "values") else y_val
            fit_kwargs["eval_set"] = Pool(X_v, y_v)
            fit_kwargs["early_stopping_rounds"] = 40

        self._model.fit(X_tr, y_tr, **fit_kwargs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        preds = self._model.predict(X_arr)
        return np.asarray(preds).ravel().astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        return self._model.predict_proba(X_arr)

    def get_feature_importances(self, feature_names: list[str]) -> pd.Series:
        """Return native CatBoost feature importances."""
        importances = self._model.get_feature_importance()
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def get_sklearn_estimator(self):
        return self._model

    def get_scaler(self):
        return self._scaler
