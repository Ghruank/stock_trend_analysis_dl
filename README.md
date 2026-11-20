# Stock Transformers: LSTM vs Transformer vs Informer

This deep-learning project compares three sequence architectures, **LSTM**, **Transformer (with Time2Vec)** and **Informer (ProbSparse attention)**, for forecasting the next 5 daily returns of 49 assets.
The data is a frozen daily snapshot from 2005 to November 2026. Models are tuned on 2022–2023 and evaluated once on **January 2024 – November 2026**.

| | |
|---|---|
| Assets | 25 Indian large caps · 10 high-volatility Indian names · 8 US mega caps · 6 commodity instruments (gold, silver, crude, natural gas, copper, gold ETF) |
| Features | 87 stationary features in 10 families (8 basic price/volume features + 79 engineered) |
| Models | LSTM with temporal attention · Transformer encoder with Time2Vec · Informer (ProbSparse self-attention, distilling, generative decoder) |
| Protocol | chronological splits with purging, train-only scaling, 3 seeds per model plus a seed ensemble, random-walk and ridge baselines |
| Compute | data and features are built locally; training ran on a free Colab T4 GPU (data uploaded once, never fetched at run time) |

## Key findings

1. **Honest evaluation first.** Splits are chronological with purging, scalers are fitted on training data only, and every score is compared with a random-walk and a linear baseline. On price levels even "tomorrow = today" scores R² ≈ 0.99, so skill is measured on returns.
2. **With a stationary target, all three architectures tie the random walk** on price error, and their next-day *direction* is right about 50% of the time. Daily market direction is essentially unpredictable from price and volume data.
3. **Innovation: a market-neutral target.** We train on returns demeaned within each region and day ("which asset beats its peers?") and early-stop on validation correlation. This turns near-zero skill into a consistent ranking signal for the LSTM. Its information coefficient goes from 0.0025 to **0.0174**, with a seed s.d. of only 0.002. Its market-neutral long/short book reaches a gross Sharpe ratio of **1.31** over 2024–26.
4. **Feature engineering matters.** With the market-neutral target, the LSTM on all 87 features reaches double the IC it gets from 8 basic price/volume features (0.0152 vs 0.0075, seed means). Removing *any* of the nine new feature families lowers the test IC.
5. **Which architecture is best? The LSTM.** It is best on IC, beat-the-peers accuracy and long/short Sharpe, it is the most stable across seeds, and it trains about 8× faster than the Informer. The Transformer learns a weaker, seed-dependent signal (IC 0.012). The Informer's long-sequence machinery does not help at a 96-day window, and it learns no ranking (IC −0.005). Averaging all three is *worse* than the LSTM alone.
6. **Calibrated forecast cones.** Every 5-day forecast comes with an 80% and a 95% range, calibrated on 2022–23 only. On the test period the real price lands inside the 80% band 79.8% of the time and inside the 95% band 94.8% of the time (typical 80% range over 5 days: ±4.6%). See `report/figures/fig_cone.png` and the website's Prediction explorer.
7. **A linear model is a strong baseline.** Ridge on the same 87 features still ranks better than every deep network (IC 0.030). At a daily horizon most of the signal is linear, so the target and the features matter more than the architecture.

## Results

<!-- RESULTS:START -->
**Raw volatility-normalised return target, 87 features, 3-seed ensemble (test Jan 2024 – Nov 2026)**

| Model | DA % | Peer-DA % | IC | R²oos % | MAPE % | L/S Sharpe | Dir. Sharpe net |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 49.92 | 49.85 | 0.0025 | -0.161 | 1.378 | 0.29 | -0.21 |
| Transformer | 50.20 | 50.16 | 0.0026 | -0.111 | 1.378 | 0.56 | 0.10 |
| Informer | 50.23 | 49.88 | -0.0083 | -0.003 | 1.377 | -0.66 | -0.03 |
| Ridge (linear) | 50.36 | 50.64 | 0.0219 | -0.609 | 1.382 | 1.75 | -0.34 |
| Naive random walk | -- | -- | 0.0000 | 0.000 | 1.376 | 0.00 | 0.00 |

**Market-neutral target (innovation), 87 features, 3-seed ensemble**

| Model | DA % | Peer-DA % | IC | R²oos % | MAPE % | L/S Sharpe | Dir. Sharpe net |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 50.51 | 50.54 | 0.0174 | -0.194 | 1.381 | 1.31 | 0.77 |
| Transformer | 49.46 | 50.04 | 0.0116 | -0.047 | 1.377 | 0.70 | -0.96 |
| Informer | 49.11 | 49.62 | -0.0054 | -0.190 | 1.378 | 0.06 | -1.42 |
| **3-model ensemble** | 49.31 | 50.19 | 0.0048 | -0.186 | 1.379 | 0.43 | -0.80 |
| Ridge (linear) | 50.31 | 50.66 | 0.0301 | -0.464 | 1.381 | 1.85 | -1.12 |
<!-- RESULTS:END -->

Metric definitions:
- **DA:** next-day sign accuracy.
- **Peer-DA:** accuracy on whether the asset beats or lags its regional peer average.
- **IC:** mean daily cross-sectional Spearman correlation.
- **R²oos:** out-of-sample R² against the zero-return forecast.
- **MAPE:** next-day close error.
- **L/S Sharpe:** annualised, gross, top 20% minus bottom 20% within each region.
- **Dir. Sharpe net:** per-asset sign strategy after 10 bp costs.

## Repository layout

```
finalproject/
├── data/                      frozen snapshot + engineered dataset (dataset.npz, scaler.npz, assets.csv, …)
├── src/stockdl/               shared library
│   ├── features.py            cleaning, 87 features, market context, cross-sectional ranks
│   ├── data.py                GPU sliding-window sampler, targets (raw / market-neutral), feature sets
│   ├── metrics.py             price, return, ranking and trading metrics + breakdowns
│   └── train.py               AdamW + warm-up/cosine, AMP, early stopping
├── LSTM/                      model code, notebook, README
├── Transformer/               model code, notebook, README, Vaswani 2017 + Time2Vec papers
├── Informer/                  model code, notebook, README, Zhou 2021 paper
├── scripts/
│   ├── download_data.py       one-off download of the frozen snapshot
│   ├── prepare_dataset.py     cleaning, features, splits, scaling -> data/dataset.npz
│   ├── ridge_study.py         linear baseline + feature-family ablation
│   ├── export_web_data.py     results -> website/public/data
│   ├── make_figures.py        figures for report and slides
│   ├── make_report_tables.py  LaTeX tables + this README's tables
│   └── make_notebooks.py      regenerates the three notebooks
├── run_experiments.py         baselines, search, final (3 seeds), market-neutral, ablation
├── live_predict.py            live 5-day forecast from the latest prices
├── results/                   runs.jsonl, predictions and checkpoints per model
├── website/                   interactive React (Vite) dashboard
├── report/                    LaTeX report (main.tex -> main.pdf), figures/, generated/ tables
└── presentation/              slide deck (Stock_Transformers.pptx) + its generator build_deck.js
```

## Reproduce

```bash
uv venv && uv pip install torch pandas numpy scipy scikit-learn yfinance pyarrow matplotlib nbformat nbconvert ipykernel
python scripts/download_data.py               # once; the snapshot is already in data/
python scripts/prepare_dataset.py             # -> data/dataset.npz
python scripts/ridge_study.py                 # linear baselines (CPU, seconds)
# on a GPU (Colab T4 is enough):
python run_experiments.py --phase baselines --out results/baselines
python run_experiments.py --phase search  --models lstm --out results/lstm      # same for transformer / informer
python run_experiments.py --phase final   --models lstm --out results/lstm
python run_experiments.py --phase xs      --models lstm --out results/lstm --xs_features full,base
python run_experiments.py --phase ablation --ablation_target xs --models lstm --out results/lstm
# reports
python scripts/export_web_data.py && python scripts/make_figures.py && python scripts/make_report_tables.py
python scripts/backfill_metrics.py   # only needed for runs made before DA_xs existed
cd report && pdflatex main.tex && pdflatex main.tex && cd ..
npm install pptxgenjs && node presentation/build_deck.js   # rebuilds the slide deck from the exported results
cd website && npm install && npm run dev       # dashboard on http://localhost:5173
```

## Live demo

```bash
python live_predict.py --refresh     # downloads today's prices, rebuilds features, runs all three checkpoints
python scripts/export_web_data.py    # pushes the forecast to the website's "Live demo" tab
```

*This is a research and education project, not investment advice.*
