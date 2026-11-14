"""Turn the frozen raw snapshot into a model-ready, fully feature-engineered dataset.

Run locally (no network needed):   python scripts/prepare_dataset.py
Output: data/dataset.npz  (+ data/dataset_meta.json, data/panel_features.parquet)

Splits (chronological, no shuffling, purged by the forecast horizon):
  train : 2007-01-01 .. 2021-12-31
  val   : 2022-01-01 .. 2023-12-31
  test  : 2024-01-01 .. latest available (2026-11)
Scaling: per ticker, mean/std estimated on the TRAIN period only, then clipped at +-5 sd.
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.features import build_panel, feature_columns  # noqa: E402

H = 5
TRAIN_START, VAL_START, TEST_START = "2007-01-01", "2022-01-01", "2024-01-01"


def main():
    prices = pd.read_csv(os.path.join(ROOT, "data", "prices.csv"))
    market = pd.read_csv(os.path.join(ROOT, "data", "market.csv"), index_col=0)
    panel = build_panel(prices, market, horizon=H)
    panel = panel[panel["date"] >= TRAIN_START].sort_values(["ticker", "date"]).reset_index(drop=True)
    cols = feature_columns(panel)
    print({k: len(v) for k, v in cols.items()})

    d = panel["date"]
    split = np.where(d < VAL_START, 0, np.where(d < TEST_START, 1, 2)).astype(np.int8)
    tickers = sorted(panel["ticker"].unique())
    tid = panel["ticker"].map({t: i for i, t in enumerate(tickers)}).values.astype(np.int16)

    # ---- purge: last H rows of train / val per ticker would peek into next split
    purge = np.zeros(len(panel), bool)
    for t in range(len(tickers)):
        idx = np.where(tid == t)[0]
        s = split[idx]
        for k in range(1, H + 1):
            nxt = np.r_[s[k:], np.full(k, 99)]
            purge[idx[(nxt != s) & (nxt != 99)]] = True

    # ---- per-ticker scaling on train period
    feat_all = cols["full"]
    X = panel[feat_all].values.astype(np.float64)
    Xs = np.zeros_like(X, dtype=np.float32)
    feat_mu = np.zeros((len(tickers), X.shape[1]))
    feat_sd = np.ones((len(tickers), X.shape[1]))
    for t in range(len(tickers)):
        m = tid == t
        tr = m & (split == 0)
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-8
        feat_mu[t], feat_sd[t] = mu, sd
        Xs[m] = np.clip((X[m] - mu) / sd, -5, 5)
    nan_frac = np.isnan(Xs).mean(0)
    Xs = np.nan_to_num(Xs, nan=0.0)

    y_ret = panel[[f"y_ret{k}" for k in range(1, H + 1)]].values.astype(np.float32)
    y_close = panel[[f"y_close{k}" for k in range(1, H + 1)]].values.astype(np.float32)
    sigma = panel["sigma"].values.astype(np.float32)
    close = panel["close"].values.astype(np.float32)
    # row validity: target exists, sigma exists, enough warm-up for long features
    warm = panel.groupby("ticker").cumcount().values >= 260
    valid = (~np.isnan(y_ret).any(1)) & (~np.isnan(sigma)) & (sigma > 0) & warm & (~purge)
    # time marks for Informer: [month, day, weekday] (0-based ints)
    marks = np.stack([d.dt.month.values - 1, d.dt.day.values - 1, d.dt.day_of_week.values], 1).astype(np.int8)

    base_idx = [feat_all.index(c) for c in cols["base"]]
    # asset metadata + volatility tier (tertiles of TRAIN-period annualised volatility)
    info = panel.groupby("ticker").agg(asset_class=("asset_class", "first"), region=("region", "first"))
    tr_vol = panel[split == 0].groupby("ticker")["B_ret1"].std() * np.sqrt(252)
    info["train_ann_vol"] = tr_vol
    q1, q2 = tr_vol.quantile([1 / 3, 2 / 3])
    info["vol_tier"] = np.where(tr_vol <= q1, "low", np.where(tr_vol <= q2, "mid", "high"))
    info = info.loc[tickers]
    info.to_csv(os.path.join(ROOT, "data", "assets.csv"))
    print(info.sort_values("train_ann_vol").to_string())
    # per-ticker scaler (train-period stats) -> needed for live inference on fresh data
    np.savez(os.path.join(ROOT, "data", "scaler.npz"), mu=feat_mu, sd=feat_sd, tickers=np.array(tickers),
             features=np.array(feat_all))
    np.savez_compressed(
        os.path.join(ROOT, "data", "dataset.npz"),
        X_full=Xs.astype(np.float16), base_idx=np.array(base_idx), y_ret=y_ret, y_close=y_close,
        sigma=sigma, close=close, split=split, valid=valid, ticker_id=tid, marks=marks,
        dates=d.values.astype("datetime64[D]").astype(np.int64),
    )
    meta = {"features_full": feat_all, "features_base": cols["base"],
            "tickers": tickers, "horizon": H,
            "asset_class": info["asset_class"].tolist(), "region": info["region"].tolist(),
            "vol_tier": info["vol_tier"].tolist(), "train_ann_vol": info["train_ann_vol"].round(4).tolist(),
            "splits": {"train": [TRAIN_START, VAL_START], "val": [VAL_START, TEST_START], "test": [TEST_START, str(d.max().date())]},
            "rows_valid_per_split": {s: int(((split == i) & valid).sum()) for i, s in enumerate(["train", "val", "test"])},
            "nan_fraction_before_fill": dict(zip(feat_all, map(float, nan_frac.round(4))))}
    json.dump(meta, open(os.path.join(ROOT, "data", "dataset_meta.json"), "w"), indent=1)
    if _has_pq():
        panel[["date", "ticker", "region", "asset_class"] + feat_all + ["sigma", "close"]].to_parquet(
            os.path.join(ROOT, "data", "panel_features.parquet"))
    print(meta["rows_valid_per_split"], Xs.shape)
    print("top NaN features:", sorted(meta["nan_fraction_before_fill"].items(), key=lambda x: -x[1])[:6])


def _has_pq():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
