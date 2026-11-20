"""FT-Transformer (Feature Tokenizer + Transformer) architecture for tabular data in PyTorch.

Reference architecture:
Gorishniy et al., 'Revisiting Deep Learning Models for Tabular Data', NeurIPS 2021.
https://arxiv.org/abs/2106.11959
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from .base_model import BaseModel


class FTTransformerNet(nn.Module):
    """PyTorch Module for Feature Tokenizer + Transformer."""

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * 2),
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens_with_cls = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(tokens_with_cls)
        cls_out = encoded[:, 0, :]
        return self.head(cls_out)


class FTTransformerModel(BaseModel):
    """
    PyTorch implementation of Feature Tokenizer + Transformer for binary classification.
    
    Includes:
    - Numerical feature embedding
    - Multi-head self-attention encoder layers with Pre-LayerNorm
    - CLS token pooling & classification head
    - AdamW optimizer with Cosine Annealing learning rate schedule
    - Early stopping on validation loss/AUC
    """

    def __init__(
        self,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 150,
        patience: int = 20,
        random_state: int = 42,
    ):
        super().__init__(name="FT-Transformer", random_state=random_state)
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dropout = ffn_dropout
        self.attention_dropout = attention_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self._scaler = StandardScaler()
        self._net: Optional[FTTransformerNet] = None
        self.train_history_: dict = {"train_loss": [], "val_loss": []}
        self.device_str: str = "cpu"

    def _build_model(self):
        return None

    def _fit_impl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = str(device)

        X_tr = self._scaler.fit_transform(X_train).astype(np.float32)
        y_tr = (y_train.values if hasattr(y_train, "values") else y_train).astype(np.int64)
        n_features = X_tr.shape[1]

        self._net = FTTransformerNet(
            n_features=n_features,
            d_token=self.d_token,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
        ).to(device)

        # Imbalance weighting
        counts = np.bincount(y_tr)
        weights = torch.tensor([len(y_tr) / (2.0 * max(c, 1)) for c in counts], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        dataset = TensorDataset(torch.from_numpy(X_tr).to(device), torch.from_numpy(y_tr).to(device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = self._scaler.transform(X_val).astype(np.float32)
            y_v = (y_val.values if hasattr(y_val, "values") else y_val).astype(np.int64)
            X_val_t = torch.from_numpy(X_v).to(device)
            y_val_t = torch.from_numpy(y_v).to(device)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self._net(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= len(y_tr)
            self.train_history_["train_loss"].append(epoch_loss)

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_out = self._net(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                self.train_history_["val_loss"].append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.to(device)

    def _get_logits(self, X: pd.DataFrame) -> np.ndarray:
        device = torch.device(self.device_str)
        self._net.eval()
        X_arr = self._scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_arr).to(device)
        with torch.no_grad():
            logits = self._net(X_t).cpu().numpy()
        return logits

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logits = self._get_logits(X)
        return np.argmax(logits, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = self._get_logits(X)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_net(self):
        return self._net

    def get_scaler(self):
        return self._scaler
