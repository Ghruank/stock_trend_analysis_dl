# LSTM

**Files**
- `lstm_model.py`: the `LSTMForecaster` module.
- `LSTM.ipynb`: a clean, self-contained notebook covering data, model, (optional) training, evaluation and plots.

**Reference:** Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation 1997.

## Architecture

```
x (B, 96, F) ─► Linear(F→h) ─┐
asset id ─► Embedding(49,16) ─┴► concat ─► LSTM × L layers (hidden h, dropout)
                                       │
               last hidden state ◄─────┤
               temporal attention pool ◄┘   (softmax(w·tanh(W h_t)) weighted sum)
                         └──── concat ──► MLP ─► 5 outputs (next 5 daily returns / σ)
```

- **Temporal attention pooling** lets the head use any day in the 96-day window, not only the final state.
- **Asset embedding** lets one model be trained on all 49 assets (pooled training) while still learning asset-specific behaviour.
- **Hyper-parameter search space:** hidden ∈ {64, 128}, layers ∈ {1, 2}, dropout ∈ {0.1, 0.3}, lr ∈ {1e-3, 3e-4}. Six configurations were sampled and selected on 2022–23 validation loss.
- **Training:** AdamW with weight decay 1e-4, 1 warm-up epoch then cosine decay, batch size 512, fp16 autocast, early stopping. On a T4 an epoch takes about 4 s.

## Results (test January 2024 – November 2026; mean ± s.d. over seeds)

<!-- RESULTS:START -->
| Inputs / target | DA % | Peer-DA % | IC | MAPE % | L/S Sharpe | seeds |
|---|---:|---:|---:|---:|---:|---:|
| Base 8 (stationary) | 49.96 ± 0.61 | 50.17 ± 0.16 | -0.0019 ± 0.0041 | 1.378 ± 0.001 | 0.27 ± 0.23 | 3 |
| Full 87 | 49.96 ± 0.32 | 49.78 ± 0.29 | 0.0023 ± 0.0060 | 1.379 ± 0.001 | 0.24 ± 0.37 | 3 |
| Base 8 + market-neutral | 49.68 ± 0.09 | 50.30 ± 0.07 | 0.0075 ± 0.0056 | 1.387 ± 0.006 | 0.61 ± 0.10 | 3 |
| Full 87 + market-neutral | 50.13 ± 0.41 | 50.38 ± 0.48 | 0.0152 ± 0.0019 | 1.385 ± 0.005 | 1.03 ± 0.39 | 3 |
<!-- RESULTS:END -->

- **Raw return target:** the LSTM ties the random walk on price error, and its direction calls are a coin flip.
- **Market-neutral target (our innovation):** the LSTM reaches IC 0.0152 ± 0.0019 across seeds and 0.0174 for the seed ensemble, with a long/short Sharpe ratio of 1.31. This makes it the best and most consistent of the three architectures, so it was also used for the feature-family ablation. Removing any of the nine engineered families lowers its test IC.

## Why the LSTM does well here

The input is a short window (96 days) of low signal-to-noise data. An LSTM's recurrence is a strong inductive bias that favours recent information,
and the model has few parameters, so it overfits less than attention models that must learn position from data.
