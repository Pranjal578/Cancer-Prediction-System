"""
Optional ensemble experiment combining all four models:
1. Soft Voting Classifier (weighted average of predicted probabilities)
2. Stacking Classifier (logistic regression meta-learner trained on out-of-fold predictions)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


class SoftVotingEnsemble:
    """Soft voting classifier combining predicted probabilities from multiple models."""

    def __init__(self, models: dict[str, Any], weights: list[float] | None = None):
        self.models = models
        self.weights = weights

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.models.values()]
        if self.weights:
            w = np.array(self.weights) / sum(self.weights)
            weighted_prob = sum(p * weight for p, weight in zip(probas, w))
            return weighted_prob
        return np.mean(probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)


class StackingEnsemble:
    """Stacking classifier trained on out-of-fold predictions to prevent meta-learner leakage."""

    def __init__(self, models: dict[str, Any], random_state: int = 42):
        self.models = models
        self.meta_learner = LogisticRegression(random_state=random_state)
        self.random_state = random_state

    def fit_meta(self, oof_probs_dict: dict[str, list[float]], y_train: pd.Series) -> StackingEnsemble:
        """Fit meta-learner strictly on out-of-fold probability vectors."""
        X_meta = np.column_stack([oof_probs_dict[m] for m in self.models.keys()])
        self.meta_learner.fit(X_meta, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        X_meta = np.column_stack(probas)
        return self.meta_learner.predict_proba(X_meta)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
