"""LSTM forecaster.

Input projection + learned asset embedding -> stacked LSTM -> [last hidden state ||
temporal-attention pooling] -> LayerNorm -> MLP head that predicts the next H (=5)
volatility-normalised daily returns jointly.
"""
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, n_assets: int, horizon: int = 5, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2, asset_emb: int = 16, **_):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout))
        self.asset = nn.Embedding(n_assets, asset_emb)
        self.asset_proj = nn.Linear(asset_emb, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, horizon))

    def forward(self, batch):
        x = self.inp(batch["x"]) + self.asset_proj(self.asset(batch["asset"]))[:, None, :]
        h, _ = self.lstm(x)                                  # (B, L, hidden)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1)   # temporal attention weights
        pooled = torch.einsum("bl,bld->bd", w, h)
        z = self.norm(torch.cat([h[:, -1], pooled], -1))
        return self.head(z)
