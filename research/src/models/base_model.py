"""Abstract base model interface for the research framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import time


class BaseModel(ABC):
    """Abstract interface that all research models (TabPFN, CatBoost, FT-Transformer, XGBoost) implement."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self._model = None
        self._train_time: float = 0.0
        self._is_fitted: bool = False

    @abstractmethod
    def _build_model(self) -> any:
        """Instantiate the underlying model object."""
        ...

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> BaseModel:
        """Fit the model and record training duration."""
        if self._model is None:
            self._model = self._build_model()
        t0 = time.time()
        self._fit_impl(X_train, y_train, X_val, y_val)
        self._train_time = time.time() - t0
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_impl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Model-specific training implementation."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels (0=Benign, 1=Malignant)."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities: array of shape (n_samples, 2)."""
        ...

    def get_train_time(self) -> float:
        """Return the training time in seconds."""
        return self._train_time

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        """Load serialized model from disk."""
        return joblib.load(path)
