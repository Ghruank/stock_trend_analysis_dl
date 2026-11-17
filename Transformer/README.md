# Transformer (encoder + Time2Vec)

**Files**
- `transformer_model.py`: the `Time2Vec` and `TransformerForecaster` modules.
- `Transformer.ipynb`: a clean notebook covering data, model, (optional) training, evaluation and plots.
- `Attention_Is_All_You_Need_Vaswani2017.pdf`: the Transformer paper.
- `Time2Vec_Kazemi2019.pdf`: the learnable time-encoding paper.

## Architecture

```
x (B, 96, F) ─► Linear(F→d) + Time2Vec(t) + asset embedding
             ─► N × pre-LN encoder block [ LayerNorm ─► Multi-Head Self-Attention ─► + ─► LayerNorm ─► FFN ─► + ]
             ─► LayerNorm ─► [ last token ‖ mean over time ] ─► MLP ─► 5 outputs
```

- **Time2Vec** (Kazemi et al., 2019) replaces the fixed sinusoidal encoding with a learnable one: one linear term ωᵢt+φᵢ plus k periodic terms sin(ωᵢt+φᵢ).
  This lets the model learn its own seasonalities, such as weekly or monthly patterns.
- **Pre-LayerNorm** blocks train stably with warm-up and no learning-rate tuning tricks.
- **Search space:** d_model ∈ {64, 128}, layers ∈ {2, 3}, dropout ∈ {0.1, 0.3}, lr ∈ {3e-4, 1e-4}, 4 heads, d_ff = 256.
- On a T4 an epoch takes about 17 s.

### Evaluation protocol

Splits are chronological with a 5-day purge gap, so overlapping windows never cross from training into test.
Scores are measured on returns, not on price levels: on price levels, any model that copies today's price looks excellent (R² ≈ 0.99).

## Results (test January 2024 – November 2026; mean ± s.d. over seeds)

<!-- RESULTS:START -->
| Inputs / target | DA % | Peer-DA % | IC | MAPE % | L/S Sharpe | seeds |
|---|---:|---:|---:|---:|---:|---:|
| Base 8 (stationary) | 50.50 ± 0.21 | 50.01 ± 0.40 | -0.0078 ± 0.0012 | 1.378 ± 0.001 | -0.07 ± 0.35 | 3 |
| Full 87 | 50.22 ± 0.47 | 50.05 ± 0.10 | 0.0001 ± 0.0064 | 1.378 ± 0.000 | 0.15 ± 0.08 | 3 |
| Full 87 + market-neutral | 49.58 ± 0.25 | 50.04 ± 0.17 | 0.0107 ± 0.0097 | 1.377 ± 0.000 | 0.96 ± 0.41 | 3 |
<!-- RESULTS:END -->

Self-attention over 96 daily steps has no advantage over recurrence on this data. With the market-neutral target, the Transformer reaches IC 0.0107 ± 0.0097: positive on average, but five times as seed-dependent as the LSTM. Validation correlation peaks within a few epochs.
