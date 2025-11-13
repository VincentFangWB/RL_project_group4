# src/advisors/gru_advisor.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, in_dim=1, hidden=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: [B, T, 1]
        y, _ = self.gru(x)
        out = self.head(y[:, -1, :])
        return out

@torch.no_grad()
def _make_dataset(returns: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(returns)-1):
        X.append(returns[i-window:i].reshape(-1,1))
        y.append([returns[i]])
    X = np.stack(X, axis=0)  # [N,T,1]
    y = np.array(y, dtype=np.float32)  # [N,1]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def fit_predict_gru(close: pd.Series, window: int = 10, hidden: int = 32, epochs: int = 5, lr: float = 1e-3) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    ret = close.pct_change().fillna(0.0).astype(np.float32)

    X, y = _make_dataset(ret.values, window)
    if len(X) < 10:
        return pd.Series(index=close.index, data=0.0)  # not enough data

    model = GRUPredictor(in_dim=1, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(max(1, epochs)):
        idx = torch.randperm(len(X))
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

    # one-step-ahead predictions over the full series (sliding)
    model.eval()
    preds = np.zeros_like(ret.values)
    with torch.no_grad():
        for i in range(window, len(ret)):
            x = torch.tensor(ret.values[i-window:i].reshape(1,window,1), dtype=torch.float32)
            preds[i] = model(x).item()

    out = pd.Series(preds, index=ret.index).fillna(0.0)
    return out.shift(0)  # already one-step ahead formulation
