"""TabPFN-2.5 model wrapper for WDBC binary breast cancer classification.

TabPFN is a Prior-Fitted Network (PFN) transformer trained offline on millions
of synthetic datasets. At inference time, it performs in-context learning over
the training dataset without backpropagation.

API considerations:
- Standard scikit-learn interface: fit, predict, predict_proba
- Operates directly on tabular numeric features
- WDBC (569 samples, 30 features) fits easily within TabPFN constraints
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel


class TabPFNModel(BaseModel):
    """TabPFN classifier wrapper supporting in-context learning and SHAP explanations."""

    def __init__(
        self,
        n_estimators: int = 4,
        device: str = "cpu",
        random_state: int = 42,
    ):
        super().__init__(name="TabPFN", random_state=random_state)
        self.n_estimators = n_estimators
        self.device = device
        self._scaler = StandardScaler()
        self._X_train_cached = None
        self._y_train_cached = None

    def _build_model(self):
        try:
            from tabpfn import TabPFNClassifier
            try:
                return TabPFNClassifier(
                    device=self.device,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                )
            except (TypeError, ValueError):
                return TabPFNClassifier(device=self.device)
        except ImportError:
            raise ImportError(
                "tabpfn package is required for TabPFNModel. Run: pip install tabpfn"
            )

    def _fit_impl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        X_tr = self._scaler.fit_transform(X_train)
        y_tr = y_train.values if hasattr(y_train, "values") else y_train

        # Cache reference for explainer
        self._X_train_cached = X_tr
        self._y_train_cached = y_tr

        self._model.fit(X_tr, y_tr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        preds = self._model.predict(X_arr)
        return np.asarray(preds).ravel().astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self._scaler.transform(X)
        return self._model.predict_proba(X_arr)

    def get_sklearn_estimator(self):
        return self._model

    def get_scaler(self):
        return self._scaler
