"""Transformer-encoder forecaster with Time2Vec.

Linear token projection to d_model, *learnable* Time2Vec time encoding (Kazemi et al. 2019),
asset embedding, pre-LN encoder blocks with GELU, then [last token || mean pool] -> MLP head
predicting the next H volatility-normalised daily returns.
"""
import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Kazemi et al. 2019: t2v(tau)[0] = w0*tau + b0 ; t2v(tau)[i] = sin(wi*tau + bi)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, out_dim - 1)

    def forward(self, tau):                     # tau: (L, 1)
        return torch.cat([self.lin(tau), torch.sin(self.per(tau))], -1)


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, n_assets: int, horizon: int = 5, d_model: int = 128,
                 n_heads: int = 4, layers: int = 3, d_ff: int = 256, dropout: float = 0.1,
                 asset_emb: int = 16, lookback: int = 96, **_):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.t2v = Time2Vec(d_model)
        self.register_buffer("tau", torch.linspace(-1, 0, lookback)[:, None])
        self.asset = nn.Sequential(nn.Embedding(n_assets, asset_emb), nn.Linear(asset_emb, d_model))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, horizon))

    def forward(self, batch):
        x = batch["x"]
        L = x.shape[1]
        h = self.inp(x) + self.t2v(self.tau[-L:])[None] + self.asset(batch["asset"])[:, None, :]
        h = self.encoder(self.drop(h))
        z = self.norm(torch.cat([h[:, -1], h.mean(1)], -1))
        return self.head(z)
