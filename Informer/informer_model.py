"""Informer (Zhou et al., AAAI 2021) - compact, self-contained re-implementation.

Follows the official Informer2020 code (ProbSparse self-attention, self-attention distilling, generative-style decoder with a
start token, fixed temporal embeddings), with two additions so the comparison with the
LSTM / Transformer is fair: an asset embedding and a direct multi-step regression target
(next H volatility-normalised returns instead of raw scaled prices).

Shapes: x_enc (B, L, F) ; mark_enc (B, L, 3) ; x_dec (B, label_len + H, F) ; out (B, H)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """1-D circular convolution over time (kernel 3), as in the paper."""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular")
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model)
        pos = torch.arange(0, c_in).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(pos * div)
        w[:, 1::2] = torch.cos(pos * div)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Daily frequency: month, day-of-month, weekday (fixed sinusoidal tables)."""

    def __init__(self, d_model):
        super().__init__()
        self.month, self.day, self.weekday = FixedEmbedding(13, d_model), FixedEmbedding(32, d_model), FixedEmbedding(7, d_model)

    def forward(self, m):
        return self.month(m[..., 0]) + self.day(m[..., 1]) + self.weekday(m[..., 2])


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, n_assets, dropout=0.1, asset_emb=16):
        super().__init__()
        self.value = TokenEmbedding(c_in, d_model)
        self.position = PositionalEmbedding(d_model)
        self.temporal = TemporalEmbedding(d_model)
        self.asset = nn.Sequential(nn.Embedding(n_assets, asset_emb), nn.Linear(asset_emb, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mark, asset):
        return self.drop(self.value(x) + self.position(x) + self.temporal(mark) + self.asset(asset)[:, None, :])


# ============================================================================ attention
class TriangularCausalMask:
    def __init__(self, B, L, device):
        self.mask = torch.triu(torch.ones([B, 1, L, L], dtype=torch.bool, device=device), diagonal=1)


class ProbMask:
    def __init__(self, B, H, L, index, scores, device):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self.mask = indicator.view(scores.shape)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super().__init__()
        self.mask_flag, self.scale = mask_flag, scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v, attn_mask=None):
        B, L, H, E = q.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, q.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", A, v).contiguous()


class ProbAttention(nn.Module):
    """ProbSparse self-attention: only the top-u 'active' queries (by the max-mean
    sparsity measurement M(q,K), estimated on a random sample of keys) get full attention;
    the 'lazy' queries receive mean(V) (or cumsum(V) when causally masked).
    Complexity O(L log L) instead of O(L^2)."""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor, self.scale, self.mask_flag = factor, scale, mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)   # sparsity measurement
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        return torch.matmul(Q_reduce, K.transpose(-2, -1)), M_top

    def _initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            return V.mean(dim=-2).unsqueeze(-2).expand(B, H, L_Q, D).clone()
        return V.cumsum(dim=-2)

    def _update_context(self, ctx, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            scores.masked_fill_(ProbMask(B, H, L_Q, index, scores, V.device).mask, -np.inf)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(ctx)
        return ctx

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        q, k, v = queries.transpose(2, 1), keys.transpose(2, 1), values.transpose(2, 1)
        U_part = min(self.factor * int(np.ceil(np.log(L_K))), L_K)
        u = min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)
        scores_top = scores_top * (self.scale or 1.0 / math.sqrt(D))
        ctx = self._initial_context(v, L_Q)
        ctx = self._update_context(ctx, v, scores_top, index, L_Q)
        return ctx.transpose(2, 1).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, mix=False):
        super().__init__()
        d_k = d_model // n_heads
        self.inner = attention
        self.q, self.k, self.v = (nn.Linear(d_model, d_k * n_heads) for _ in range(3))
        self.o = nn.Linear(d_k * n_heads, d_model)
        self.n_heads, self.mix = n_heads, mix

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        out = self.inner(self.q(queries).view(B, L, H, -1), self.k(keys).view(B, S, H, -1),
                         self.v(values).view(B, S, H, -1), attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        return self.o(out.view(B, L, -1))


# ============================================================================ encoder / decoder
class ConvLayer(nn.Module):
    """Self-attention distilling: Conv1d -> BatchNorm -> ELU -> MaxPool(stride 2) halves L."""

    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_in, 3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(c_in)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.pool(self.act(self.norm(self.conv(x.permute(0, 2, 1)))))
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.attention = attention
        self.conv1, self.conv2 = nn.Conv1d(d_model, d_ff, 1), nn.Conv1d(d_ff, d_model, 1)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = self.norm1(x + self.drop(self.attention(x, x, x, attn_mask)))
        y = self.drop(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        if self.conv_layers is not None:
            for attn, conv in zip(self.attn_layers, self.conv_layers):
                x = conv(attn(x))
            x = self.attn_layers[-1](x)
        else:
            for attn in self.attn_layers:
                x = attn(x)
        return self.norm(x) if self.norm is not None else x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention, self.cross_attention = self_attention, cross_attention
        self.conv1, self.conv2 = nn.Conv1d(d_model, d_ff, 1), nn.Conv1d(d_ff, d_model, 1)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        x = self.norm1(x + self.drop(self.self_attention(x, x, x)))
        x = self.norm2(x + self.drop(self.cross_attention(x, cross, cross)))
        y = self.drop(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.drop(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        return self.norm(x) if self.norm is not None else x


class InformerForecaster(nn.Module):
    def __init__(self, n_features: int, n_assets: int, horizon: int = 5, d_model: int = 128,
                 n_heads: int = 4, e_layers: int = 2, d_layers: int = 1, d_ff: int = 256,
                 factor: int = 5, dropout: float = 0.1, attn: str = "prob", distil: bool = True,
                 label_len: int = 48, mix: bool = True, **_):
        super().__init__()
        self.horizon, self.label_len = horizon, label_len
        Attn = ProbAttention if attn == "prob" else FullAttention
        mk = (lambda mask: Attn(mask, factor, attention_dropout=dropout)) if attn == "prob" else \
             (lambda mask: FullAttention(mask, attention_dropout=dropout))
        self.enc_embedding = DataEmbedding(n_features, d_model, n_assets, dropout)
        self.dec_embedding = DataEmbedding(n_features, d_model, n_assets, dropout)
        self.encoder = Encoder(
            [EncoderLayer(AttentionLayer(mk(False), d_model, n_heads), d_model, d_ff, dropout) for _ in range(e_layers)],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model))
        self.decoder = Decoder(
            [DecoderLayer(AttentionLayer(mk(True), d_model, n_heads, mix=mix),
                          AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                          d_model, d_ff, dropout) for _ in range(d_layers)],
            norm_layer=nn.LayerNorm(d_model))
        self.projection = nn.Linear(d_model, 1)

    def forward(self, batch):
        x_enc, asset = batch["x"], batch["asset"]
        # decoder input = start token (last label_len encoder steps) + zero placeholders for H steps
        x_dec = torch.cat([x_enc[:, -self.label_len:], torch.zeros_like(x_enc[:, : self.horizon])], 1)
        enc = self.encoder(self.enc_embedding(x_enc, batch["mark_enc"], asset))
        dec = self.decoder(self.dec_embedding(x_dec, batch["mark_dec"], asset), enc)
        return self.projection(dec)[:, -self.horizon:, 0]
