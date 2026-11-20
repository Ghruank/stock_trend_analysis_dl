"""Generate the three clean, self-contained architecture notebooks (LSTM/, Transformer/, Informer/).

python scripts/make_notebooks.py && (cd LSTM && jupyter nbconvert --to notebook --execute --inplace LSTM.ipynb) ...
"""
import os

import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INTRO = {
    "lstm": """# LSTM — next-day / 5-day stock return forecasting

**Stock Transformers** · Deep Learning final project · Khush Agrawal, Yashvi Mehta, Ghruank Kothare

### Architecture
```
x (96 days × 87 features) ──► Linear+GELU (d=hidden) ──(+ asset embedding)──► LSTM × layers
        └──► [ last hidden state ‖ temporal-attention pooled state ] ──► LayerNorm ──► MLP ──► 5 outputs
```
* **Why LSTM:** gated recurrence keeps a memory cell, so it can pick up both short-term (days) and slower (months) regimes.
* **Temporal attention pooling** lets the network weigh informative days in the 96-day window instead of relying only on the last state.
* **Output:** the next 5 daily log-returns divided by the asset's current EWMA volatility σₜ (a stationary, unit-free target).

""",
    "transformer": """# Transformer (+ Time2Vec) — next-day / 5-day stock return forecasting

**Stock Transformers** · Deep Learning final project · Khush Agrawal, Yashvi Mehta, Ghruank Kothare

Papers in this folder: *Attention Is All You Need* (Vaswani et al., 2017) and *Time2Vec* (Kazemi et al., 2019).

### Architecture
```
x (96 × 87) ──► Linear (d_model) ──(+ learnable Time2Vec(τ) + asset embedding)──►
   [ Pre-LN encoder block: MultiHeadSelfAttention → +res → LN → FFN(GELU) → +res ] × layers
   ──► [ last token ‖ mean over time ] ──► LayerNorm ──► MLP ──► 5 outputs
```
* **Self-attention** compares every day in the window with every other day in parallel — long-range dependencies without recurrence.
* **Time2Vec** (Kazemi et al., 2019) is a *trainable* time encoding: one linear + (d−1) sinusoidal components of time, added like a positional encoding.
* **Pre-LayerNorm** blocks train stably at small batch sizes and modest depth.

""",
    "informer": """# Informer — next-day / 5-day stock return forecasting

**Stock Transformers** · Deep Learning final project · Khush Agrawal, Yashvi Mehta, Ghruank Kothare

Paper in this folder: *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* (Zhou et al., AAAI 2021, best paper).

### Architecture
```
Encoder input  : x (96 × 87)  ─► Conv1d token emb + positional + temporal(month, day, weekday) + asset emb
                 [ ProbSparse self-attention block ─► distilling Conv1d+ELU+MaxPool(÷2) ] × (e_layers−1) ─► last block
Decoder input  : last 48 days of x (start token) ‖ 5 zero placeholders, same embeddings
                 masked ProbSparse self-attention ─► cross-attention to encoder ─► FFN
                 ─► Linear ─► 5 outputs (generative-style: all horizons in one forward pass)
```
* **ProbSparse attention:** only the top-u = c·ln L "active" queries (largest max–mean score gap, estimated on a random key sample) get full attention; "lazy" queries receive mean(V). Cost O(L log L) instead of O(L²).
* **Self-attention distilling** halves the sequence between encoder blocks, keeping dominant features and cutting memory.
* **Generative decoder** predicts the whole horizon at once — no error accumulation from step-by-step decoding.

""",
}

SETUP = '''import os, sys, json, glob
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

# ---- locate the project root (works locally and on Colab after unpacking the bundle into /content/proj)
ROOT = os.path.abspath("..") if os.path.exists("../src/stockdl") else "/content/proj"
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.data import load_panel, feature_group
from stockdl.train import fit, predict, build_model
from stockdl.metrics import evaluate, naive_metrics, breakdown

MODEL = "{model}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = False          # True = retrain the selected configuration here (GPU recommended, ~5-30 min)
RESULTS = os.environ.get("STOCKDL_RESULTS", os.path.join(ROOT, "results"))
plt.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "font.size": 10}})
COLORS = {{"lstm": "#2a78d6", "transformer": "#eb6834", "informer": "#1baf7a", "ridge": "#898781", "naive": "#898781"}}
print("root:", ROOT, "| device:", DEVICE)'''

DATA = '''panel = load_panel(os.path.join(ROOT, "data", "dataset.npz"), "full", lookback=96, device="cpu")
meta = panel.meta
print(f"assets: {len(meta['tickers'])}  features: {panel.X.shape[1]}  horizon: {panel.horizon} days")
for s in ("train", "val", "test"):
    print(f"{s:5s} windows: {len(panel.ends[s]):7d}   period: {meta['splits'][s]}")
groups = pd.Series([feature_group(n) for n in meta["features_full"]]).value_counts()
groups.plot.barh(color="#2a78d6", figsize=(6, 3), title="Engineered features per family"); plt.show()'''

MODEL_CELL = '''rows = [json.loads(l) for l in open(os.path.join(RESULTS, MODEL, "runs.jsonl"))]
search = [r for r in rows if r["phase"] == "search"]
best = min(search, key=lambda r: r["best_val_loss"])
cfg = best["cfg"]
print("hyper-parameter search (validation 2022-2023):")
display(pd.DataFrame([{**r["cfg"], "val_loss": r["best_val_loss"], "val_DA_%": r["val"]["DA_%"], "epochs": r["epochs_run"],
                       "params": r["n_params"]} for r in search]).round(4))
print("selected:", cfg)
net = build_model(MODEL, panel, cfg)
print(net)
print(f"trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")'''

TRAIN_CELL = '''if TRAIN:
    p = load_panel(os.path.join(ROOT, "data", "dataset.npz"), "full", 96, DEVICE)
    net, info = fit(MODEL, p, cfg, seed=0)
    pred = predict(net, p, "test")
    met, _ = evaluate(pred, p.ends["test"], p)
    print(pd.Series(met).round(4))
else:
    print("TRAIN=False -> using the runs trained on the Colab T4 GPU (results/%s/runs.jsonl)" % MODEL)

# learning curves of the selected configuration's final runs
fin = [r for r in rows if r["phase"] == "final" and r["features"] == "full"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for r in fin:
    h = pd.DataFrame(r["history"])
    ax[0].plot(h.epoch + 1, h.train_loss, color=COLORS[MODEL], alpha=.4 + .2 * r["seed"], label=f"seed {r['seed']}")
    ax[1].plot(h.epoch + 1, h.val_loss, color=COLORS[MODEL], alpha=.4 + .2 * r["seed"], label=f"seed {r['seed']}")
ax[0].set_title("train loss (MSE, vol-normalised returns)"); ax[1].set_title("validation loss")
for a in ax: a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.show()'''

EVAL_CELL = '''KEY = ["DA_%", "DA_xs_%", "IC", "R2_oos_%", "MAPE_%", "Sharpe_LS", "Sharpe_dir_net", "DA_5d_%"]
tab = []
for phase, ens_phase, fs, label in [("final", "ensemble", "base", "base"),
                                    ("final", "ensemble", "full", "full"), ("xs", "ensemble_xs", "base", "market-neutral base"),
                                    ("xs", "ensemble_xs", "full", "market-neutral full")]:
    rr = [r for r in rows if r["phase"] == phase and r["features"] == fs]
    if rr:
        d = pd.DataFrame([r["test"] for r in rr]).reindex(columns=KEY)
        tab.append(pd.Series({k: f"{d[k].mean():.3f} ± {d[k].std():.3f}" if len(d) > 1 else f"{d[k].mean():.3f}" for k in KEY}, name=f"{label} ({len(rr)} seeds)"))
    e = [r for r in rows if r["phase"] == ens_phase and r["features"] == fs]
    if e:
        tab.append(pd.Series({k: f"{e[0]['test'].get(k, float('nan')):.3f}" for k in KEY}, name=f"{label} seed-ensemble"))
base_rows = [json.loads(l) for l in open(os.path.join(RESULTS, "baselines", "runs.jsonl"))]
for r in base_rows:
    if "key" in r and not r["key"].startswith("baseline"):
        continue
    tab.append(pd.Series({k: ("—" if r["test"].get(k) is None or (isinstance(r["test"][k], float) and np.isnan(r["test"][k])) else f"{r['test'][k]:.3f}") for k in KEY}, name=f"baseline: {r['model']}"))
print("TEST 2024-01 .. 2026-11 (out-of-sample)")
display(pd.DataFrame(tab))'''

PLOT_CELL = '''P = panel.np_
f = os.path.join(RESULTS, MODEL, "preds", f"ensemble_{MODEL}_full.npz")
if not os.path.exists(f):
    f = os.path.join(RESULTS, MODEL, "preds", f"final_{MODEL}_full_s0.npz")
z = np.load(f)
pred, te = z["pred"], z["ends"]
r_hat = pred[:, 0] * P["sigma"][te]
close_t, close_next = P["close"][te], P["y_close"][te, 0]
dates = pd.to_datetime(P["dates"][te], unit="D") + pd.offsets.BDay(1)
fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
for ax, t in zip(axes, ["TCS", "GOLD", "TSLA"]):
    m = P["tid"][te] == meta["tickers"].index(t)
    ax.plot(dates[m][-150:], close_next[m][-150:], color="#0b0b0b", lw=1.6, label="actual close")
    ax.plot(dates[m][-150:], (close_t * np.exp(r_hat))[m][-150:], color=COLORS[MODEL], lw=1.2, label=f"{MODEL} forecast")
    ax.set_title(f"{t} — next-day close, last 150 test days"); ax.legend(loc="upper left")
plt.tight_layout(); plt.show()

y = P["y_ret"][te, 0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hexbin(r_hat * 100, y * 100, gridsize=60, bins="log", cmap="Blues", extent=(-3, 3, -8, 8))
ax[0].axhline(0, color="#c3c2b7"); ax[0].axvline(0, color="#c3c2b7")
ax[0].set_xlabel("predicted next-day return %"); ax[0].set_ylabel("actual %"); ax[0].set_title("forecast vs outcome (all assets)")
# calibration: average realised return per forecast decile
q = pd.qcut(r_hat, 10, labels=False, duplicates="drop")
dec = pd.Series(y).groupby(q).mean() * 1e4
ax[1].bar(dec.index + 1, dec.values, color=[COLORS[MODEL] if v > 0 else "#c3c2b7" for v in dec.values])
ax[1].set_xlabel("forecast decile (1 = most bearish)"); ax[1].set_ylabel("mean realised return (bps)")
ax[1].set_title("monotonic deciles = useful ranking signal")
plt.tight_layout(); plt.show()'''

XS_CELL = '''# Raw target vs market-neutral target: daily ranking quality on the test set
def quintiles(fname):
    z = np.load(os.path.join(RESULTS, MODEL, "preds", fname))
    te = z["ends"]
    df = pd.DataFrame({"d": P["dates"][te], "g": np.array(meta["region"])[P["tid"][te]], "p": z["pred"][:, 0], "y": P["y_ret"][te, 0]})
    df["q"] = df.groupby(["d", "g"]).p.rank(pct=True)
    df["Q"] = np.minimum((df.q * 5).astype(int), 4) + 1
    return df.groupby("Q").y.mean() * 1e4
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for a, (f, t) in zip(ax, [(f"ensemble_{MODEL}_full.npz", "raw return target"), (f"ensemble_xs_{MODEL}_full.npz", "market-neutral target")]):
    if os.path.exists(os.path.join(RESULTS, MODEL, "preds", f)):
        q = quintiles(f)
        a.bar(q.index, q.values, color=[COLORS[MODEL] if v > 0 else "#c3c2b7" for v in q.values])
    a.axhline(0, color="#898781"); a.set_xticks(range(1, 6), ["Q1\\nbearish", "Q2", "Q3", "Q4", "Q5\\nbullish"])
    a.set_title(f"next-day return by daily forecast quintile (bps) — {t}")
plt.tight_layout(); plt.show()'''

CONE_CELL = '''# Forecast cone: predicted 5-day path + calibrated 80% / 95% range, then the actual price over it
from stockdl.metrics import cone_quantiles, cone_coverage
va = panel.ends["val"]
Q = cone_quantiles(P["y_ret"][va].astype(float), P["sigma"][va].astype(float))   # band width from 2022-23 only
f = os.path.join(RESULTS, MODEL, "preds", f"ensemble_{MODEL}_full.npz")
z = np.load(f if os.path.exists(f) else f.replace("ensemble_", "final_").replace("_full.npz", "_full_s0.npz"))
te, r5 = z["ends"], z["pred"] * P["sigma"][z["ends"]][:, None]
cov = cone_coverage(r5, P["y_ret"][te].astype(float), P["sigma"][te].astype(float), Q)
print(f"test coverage: 80% band {cov[80]['all']:.1f}%  |  95% band {cov[95]['all']:.1f}%   (calibrated = 80 / 95)")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, t in zip(axes, ["TCS", "GOLD", "TSLA"]):
    rows_t = np.where(P["tid"][te] == meta["tickers"].index(t))[0]
    d0 = pd.to_datetime(P["dates"][te][rows_t], unit="D")
    c = P["close"][te][rows_t]
    lo_i = len(rows_t) - 6 - 60
    ax.plot(d0[lo_i:], c[lo_i:], color="#0b0b0b", lw=1.5, label="actual close")
    for k, j in enumerate(range(lo_i, len(rows_t) - 6, 5)):
        i = rows_t[j]
        cum = np.r_[0, np.cumsum(r5[i])]
        d = d0[j:j + 6]
        for L, a in [(95, .13), (80, .3)]:
            lo, hi = Q[L]
            ax.fill_between(d, c[j] * np.exp(cum + np.r_[0, lo] * P["sigma"][te][i]), c[j] * np.exp(cum + np.r_[0, hi] * P["sigma"][te][i]),
                            color=COLORS[MODEL], alpha=a, lw=0, label=f"{L}% range" if k == 0 else None)
        ax.plot(d, c[j] * np.exp(cum), color=COLORS[MODEL], ls="--", lw=1.1, label="forecast path" if k == 0 else None)
    ax.set_title(f"{t}: weekly 5-day cones, last 3 months"); ax.tick_params(axis="x", labelsize=8, rotation=30)
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.show()'''

BREAK_CELL = '''ens = [r for r in rows if r["phase"] == "ensemble" and r["features"] == "full"]
src = ens[0] if ens else [r for r in rows if r["phase"] == "final" and r["features"] == "full"][0]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
vt = pd.DataFrame(src["by_vol_tier"]).T.loc[["low", "mid", "high"]]
ac = pd.DataFrame(src["by_asset_class"]).T
ax[0].bar(vt.index, vt["DA_%"], color=COLORS[MODEL]); ax[0].axhline(50, color="#898781", ls="--")
ax[0].set_ylim(45, max(55, vt["DA_%"].max() + 1)); ax[0].set_title("directional accuracy by volatility tier")
ax[1].bar(ac.index, ac["DA_%"], color=COLORS[MODEL]); ax[1].axhline(50, color="#898781", ls="--")
ax[1].set_ylim(45, max(55, ac["DA_%"].max() + 1)); ax[1].set_title("directional accuracy by asset class")
plt.tight_layout(); plt.show()
display(pd.concat({"volatility tier": vt, "asset class": ac}).round(3))'''

CONCL = {
    "lstm": "## Takeaways\nSee the printed test table above. The project-level comparison of all three architectures (and the feature-ablation study, which was run with this LSTM) is in the main `README.md`, the website and the report.",
    "transformer": "## Takeaways\nSee the printed test table above. The project-level comparison of all three architectures is in the main `README.md`, the website and the report.",
    "informer": "## Takeaways\nSee the printed test table above. The project-level comparison of all three architectures is in the main `README.md`, the website and the report.",
}


def make(model, folder, fname):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO[model]),
        nbf.v4.new_markdown_cell("## 0 · Setup\nOn Colab: upload the project bundle, `tar xzf` it into `/content/proj`, and run all. "
                                 "Locally: run from this folder with the project's virtualenv. Nothing is downloaded at runtime — "
                                 "the feature-engineered dataset is `data/dataset.npz` (built by `scripts/prepare_dataset.py`)."),
        nbf.v4.new_code_cell(SETUP.format(model=model)),
        nbf.v4.new_markdown_cell("## 1 · Data\n49 assets (25 Indian large caps, 10 volatile Indian mid/small caps, 8 US mega caps, 6 commodities), "
                                 "daily bars 2005 → Nov 2026. 87 engineered features; per-asset scaling fitted on the training period only."),
        nbf.v4.new_code_cell(DATA),
        nbf.v4.new_markdown_cell("## 2 · Model & hyper-parameter search\nConfigurations were scored on 2022–2023 validation loss only; the test period was never used for selection."),
        nbf.v4.new_code_cell(MODEL_CELL),
        nbf.v4.new_markdown_cell("## 3 · Training\nAdamW, 1-epoch warm-up + cosine decay, gradient clipping 1.0, mixed precision on GPU, early stopping on validation loss."),
        nbf.v4.new_code_cell(TRAIN_CELL),
        nbf.v4.new_markdown_cell("## 4 · Test-set evaluation (Jan 2024 – Nov 2026)\n"
                                 "* **DA** directional accuracy · **IC** daily cross-sectional rank correlation · **R²oos** vs. the zero-return forecast\n"
                                 "* **MAPE** next-day price error (compare with the naive random walk) · **Sharpe_LS** market-neutral long/short (gross) · **Sharpe_dir_net** sign strategy after 10 bps costs\n"
                                 "* *base* = 8 basic price/volume features (stationary) · *full* = 87 engineered features"),
        nbf.v4.new_code_cell(EVAL_CELL),
        nbf.v4.new_markdown_cell("## 5 · What the forecasts look like"),
        nbf.v4.new_code_cell(PLOT_CELL),
        nbf.v4.new_markdown_cell("## 6 · Innovation: market-neutral target\nPredicting *where the market goes* is a coin flip. Re-training on returns demeaned "
                                 "within (region, day) asks *which asset beats its peers* instead; checkpoints are chosen on validation correlation. "
                                 "A monotone staircase from Q1 to Q5 means the daily ranking is informative."),
        nbf.v4.new_code_cell(XS_CELL),
        nbf.v4.new_markdown_cell("## 7 · Forecast cones: predicted range vs what actually happened\n"
                                 "Every 5-day forecast gets an 80% and a 95% range. The width comes from how far prices moved in the 2022–23 "
                                 "validation period (in units of each asset's current volatility), so nothing from the test period is used. "
                                 "If the ranges are honest, the actual price should land inside the 80% band about 80% of the time."),
        nbf.v4.new_code_cell(CONE_CELL),
        nbf.v4.new_markdown_cell("## 8 · Volatile vs non-volatile, shares vs commodities"),
        nbf.v4.new_code_cell(BREAK_CELL),
        nbf.v4.new_markdown_cell(CONCL[model]),
    ]
    nbf.write(nb, os.path.join(ROOT, folder, fname))


if __name__ == "__main__":
    make("lstm", "LSTM", "LSTM.ipynb")
    make("transformer", "Transformer", "Transformer.ipynb")
    make("informer", "Informer", "Informer.ipynb")
    print("notebooks written")
