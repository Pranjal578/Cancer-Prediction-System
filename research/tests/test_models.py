"""Tests for research model interfaces and contracts across all 4 models."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.ft_transformer import FTTransformerModel
from src.models.tabpfn_model import TabPFNModel


def create_dummy_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(40, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.choice([0, 1], size=40))
    return X, y


def test_xgboost_contract():
    X, y = create_dummy_data()
    model = XGBoostModel(random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_catboost_contract():
    X, y = create_dummy_data()
    model = CatBoostModel(params={"iterations": 10}, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_ft_transformer_contract():
    X, y = create_dummy_data()
    model = FTTransformerModel(max_epochs=3, batch_size=16, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_tabpfn_contract():
    X, y = create_dummy_data()
    model = TabPFNModel(device="cpu", random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})

    probas = model.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
