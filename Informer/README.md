# Informer

**Files**
- `informer_model.py`: a faithful re-implementation of Zhou et al. (AAAI 2021) with ProbSparse attention, distilling and a generative decoder.
- `Informer.ipynb`: a clean notebook covering data, model, (optional) training, evaluation and plots.
- `Informer_Zhou2021_AAAI.pdf`: the paper.

## Architecture

```
Encoder input  : x (B, 96, F)  ─► token Conv1d + positional + temporal(month, day, weekday) + asset embedding
Encoder        : [ ProbSparse self-attention ─► FFN ] ─► distilling (Conv1d ─► BN ─► ELU ─► MaxPool /2) ─► …  (96 → 48 → 24)
Decoder input  : last 48 encoder steps (start token) ‖ 5 zero placeholders, with their future calendar marks
Decoder        : masked ProbSparse self-attention ─► full cross-attention ─► FFN
Head           : Linear on the 5 placeholder positions ─► next 5 daily returns / σ  (one generative forward pass)
```

- **ProbSparse attention:** for each head, only the top u = c·ln L "active" queries are scored. Queries are ranked by the max–mean sparsity measure M(q, K), and the remaining lazy queries receive the mean of V. This brings attention cost down to O(L log L).
- **Self-attention distilling** halves the sequence length after each encoder layer and keeps the dominant features.
- **Generative decoder:** all 5 horizon steps are produced at once, with no autoregressive error accumulation.
- **Search space:** d_model ∈ {64, 128}, e_layers ∈ {2, 3}, dropout ∈ {0.1, 0.3}, lr ∈ {3e-4, 1e-4}, factor c = 5, 4 heads, d_ff = 256.
  Because of the free-GPU budget, 3 configurations were evaluated.
- On a T4 an epoch takes about 33 s, the slowest of the three models.

### Target

The Informer predicts the next 5 volatility-normalised returns with exactly the same protocol as the other two models,
rather than raw prices, which drift outside the training range and cannot be compared across assets.

## Results (test January 2024 – November 2026; mean ± s.d. over seeds)

<!-- RESULTS:START -->
| Inputs / target | DA % | Peer-DA % | IC | MAPE % | L/S Sharpe | seeds |
|---|---:|---:|---:|---:|---:|---:|
| Base 8 (stationary) | 50.80 ± 0.14 | 50.17 ± 0.37 | -0.0015 ± 0.0050 | 1.378 ± 0.000 | 0.06 ± 0.55 | 3 |
| Full 87 | 50.27 ± 0.68 | 49.92 ± 0.32 | -0.0044 ± 0.0009 | 1.378 ± 0.000 | -0.25 ± 0.11 | 3 |
| Full 87 + market-neutral | 49.53 ± 0.43 | 49.75 ± 0.12 | -0.0060 ± 0.0013 | 1.379 ± 0.002 | -0.08 ± 0.23 | 3 |
<!-- RESULTS:END -->

The Informer was designed for long-sequence forecasting, with inputs of hundreds to thousands of steps and long horizons.
With a 96-day window and a 5-day horizon its efficiency tricks bring no benefit, and the sparse query sampling adds variance.
It ties the others on price error, but it is the only architecture that learns no market-neutral ranking (IC −0.006 ± 0.001), at about 8× the LSTM's training cost.
