"""Live demo: forecast the next 5 trading days for every asset with the trained models.

    python live_predict.py                      # uses the frozen snapshot in data/
    python live_predict.py --refresh            # downloads the latest prices first (needs internet)
    python live_predict.py --tickers TCS,INFY,GOLD --models informer

Pipeline (identical to training, no re-fitting of anything):
  raw OHLCV -> clean_prices -> build_panel (87 engineered features)
  -> per-ticker scaler fitted on 2007-2021 (data/scaler.npz) -> clip +-5
  -> last 96-day window -> model -> vol-normalised returns x sigma -> price path
Writes results/live_forecast.json (consumed by the website's Live Demo page).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.features import build_panel  # noqa: E402
from stockdl.train import MODELS  # noqa: E402

H = 5


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    kw = {k: v for k, v in ck["cfg"].items() if k not in ("lr", "weight_decay", "batch_size", "epochs", "patience", "monitor")}
    m = MODELS[ck["model"]](n_features=ck["n_features"], n_assets=49, horizon=H, lookback=ck["lookback"], **kw)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="download fresh prices with yfinance")
    ap.add_argument("--models", default="lstm,transformer,informer")
    ap.add_argument("--tickers", default="")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    data_dir = os.path.join(ROOT, "data")
    if a.refresh:
        data_dir = os.path.join(ROOT, "data", "live")
        os.makedirs(data_dir, exist_ok=True)
        end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "download_data.py"), "--end", end,
                        "--out", data_dir], check=True)
    prices = pd.read_csv(os.path.join(data_dir, "prices.csv"))
    market = pd.read_csv(os.path.join(data_dir, "market.csv"), index_col=0)
    sc = np.load(os.path.join(ROOT, "data", "scaler.npz"))
    tickers, feats = list(sc["tickers"]), list(sc["features"])
    want = [t for t in (a.tickers.split(",") if a.tickers else tickers) if t in tickers]
    prices = prices[prices.ticker.isin(want)]
    # cross-sectional features need the whole region, so keep everything for build_panel
    panel = build_panel(pd.read_csv(os.path.join(data_dir, "prices.csv")), market, horizon=H)

    models = {}
    for name in a.models.split(","):
        path = os.path.join(a.results, name, "models", f"{name}_full_s0.pt")
        if os.path.exists(path):
            models[name] = load_model(path)
    if not models:
        sys.exit("no checkpoints found under results/<model>/models/")
    xs_models = {}
    for name in a.models.split(","):
        path = os.path.join(a.results, name, "models", f"{name}_xs_full_s0.pt")
        if os.path.exists(path):
            xs_models[name] = load_model(path)

    out = {"generated_for": None, "assets": []}
    for t in want:
        d = panel[panel.ticker == t].sort_values("date")
        L = max(ck["lookback"] for _, ck in models.values())
        d = d.iloc[-L:]
        ti = tickers.index(t)
        X = ((d[feats].values - sc["mu"][ti]) / sc["sd"][ti]).clip(-5, 5)
        X = np.nan_to_num(X).astype(np.float32)
        dates = pd.DatetimeIndex(d["date"])
        fut = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=H)
        alld = dates.append(fut)
        marks = np.stack([alld.month - 1, alld.day - 1, alld.dayofweek], 1).astype(np.int64)
        batch = {"x": torch.tensor(X)[None], "asset": torch.tensor([ti]),
                 "mark_enc": torch.tensor(marks[:L])[None],
                 "mark_dec": torch.tensor(np.concatenate([marks[L - 48:L], marks[L:]]))[None]}
        close, sigma = float(d["close"].iloc[-1]), float(d["sigma"].iloc[-1])
        row = {"ticker": t, "last_date": str(dates[-1].date()), "last_close": close, "sigma": sigma,
               "history": [[str(x.date()), float(c)] for x, c in zip(dates[-60:], d["close"].values[-60:])],
               "future_dates": [str(x.date()) for x in fut], "forecasts": {}}
        for name, (m, _) in models.items():
            with torch.no_grad():
                r = m(batch)[0].numpy() * sigma
            path = close * np.exp(np.cumsum(r))
            row["forecasts"][name] = {"daily_ret": r.tolist(), "path": path.tolist(),
                                      "ret_5d_%": float((np.exp(r.sum()) - 1) * 100),
                                      "signal": "UP" if r.sum() > 0 else "DOWN",
                                      "z_score": float(r.sum() / (sigma * np.sqrt(H)))}
        # market-neutral models: relative-strength score (higher = expected to beat regional peers)
        if xs_models:
            sc_ = {}
            for name, (m, _) in xs_models.items():
                with torch.no_grad():
                    sc_[name] = float(m(batch)[0].numpy().sum())
            row["xs_raw"] = sc_
        out["assets"].append(row)
        out["generated_for"] = max(out["generated_for"] or "", str(dates[-1].date()))
        f = row["forecasts"]
        print(f"{t:11s} {row['last_date']} close={close:10.2f}  " +
              "  ".join(f"{k}: {v['signal']:4s} {v['ret_5d_%']:+.2f}%" for k, v in f.items()))
    # daily ranking within region from the market-neutral model(s)
    if xs_models:
        reg = dict(zip(pd.read_csv(os.path.join(ROOT, "data", "assets.csv")).ticker,
                       pd.read_csv(os.path.join(ROOT, "data", "assets.csv")).region))
        df = pd.DataFrame([{"ticker": r["ticker"], "region": reg.get(r["ticker"], "?"), **r["xs_raw"]} for r in out["assets"]])
        # rank with the best market-neutral architecture (LSTM; see README) - fall back to the average of what is available
        names = ["lstm"] if "lstm" in xs_models else list(xs_models)
        df["score"] = df.groupby("region")[names].transform(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(1)
        df["pct"] = df.groupby("region").score.rank(pct=True)
        df["action"] = np.where(df.pct > 0.8, "LONG", np.where(df.pct <= 0.2, "SHORT", "—"))
        for r in out["assets"]:
            x = df[df.ticker == r["ticker"]].iloc[0]
            r["ranking"] = {"region": x.region, "score": float(x.score), "percentile": float(x.pct), "action": x.action}
        print(df.sort_values(["region", "score"], ascending=[True, False])[["ticker", "region", "score", "pct", "action"]].to_string(index=False))
    os.makedirs(a.results, exist_ok=True)
    json.dump(out, open(os.path.join(a.results, "live_forecast.json"), "w"), indent=1)
    print("wrote", os.path.join(a.results, "live_forecast.json"))


if __name__ == "__main__":
    main()
