"""Evaluation metrics.

Price-level metrics (RMSE/MAE/MAPE/R^2 on the next-day close) are the usual way stock
forecasts are reported, but they are dominated by persistence: tomorrow's price ~= today's.
A model is only useful if it beats the naive random-walk forecast P_hat(t+1) = P(t).
We therefore report, alongside every price metric, the same metric for the naive forecast
and the return-space metrics that actually measure predictive skill:

  * R2_oos  : 1 - SSE(model returns) / SSE(zero-return forecast)   (Campbell & Thompson, 2008)
  * DA      : directional accuracy (sign of next-day return), excluding zero-return days
  * IC      : mean daily cross-sectional Spearman rank correlation (within region)
  * Sharpe  : annualised Sharpe of a daily long/short strategy using the forecasts
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def price_metrics(close_t, close_true, close_pred):
    err = close_pred - close_true
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err) / close_true) * 100),
        "R2_price": _r2(close_true, close_pred),
    }


def return_metrics(r_true, r_pred, dates, tid, region, cost_bps: float = 10.0):
    m = {}
    m["R2_oos_%"] = float((1 - np.sum((r_true - r_pred) ** 2) / np.sum(r_true ** 2)) * 100)
    nz = r_true != 0
    m["DA_%"] = float(np.mean(np.sign(r_pred[nz]) == np.sign(r_true[nz])) * 100)
    m["corr"] = float(np.corrcoef(r_true, r_pred)[0, 1]) if np.std(r_pred) > 0 else 0.0
    df = pd.DataFrame({"d": dates, "t": tid, "g": region, "y": r_true, "p": r_pred})
    ics = []
    for _, g in df.groupby(["g", "d"]):
        if len(g) >= 5 and g.p.std() > 0:
            ics.append(spearmanr(g.y, g.p)[0])
    ics = np.array(ics)
    m["IC"] = float(np.nanmean(ics)) if len(ics) else 0.0
    m["IC_IR"] = float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12) * np.sqrt(252)) if len(ics) else 0.0
    # (a) per-asset directional strategy: position = sign(forecast), equal-weight across assets
    df["pos"] = np.sign(df.p)
    df["pnl"] = df.pos * df.y
    df = df.sort_values(["t", "d"])
    df["turn"] = df.groupby("t").pos.diff().abs().fillna(1)
    df["pnl_net"] = df.pnl - df.turn * cost_bps / 1e4
    daily = df.groupby("d")[["pnl", "pnl_net"]].mean()
    for col, name in [("pnl", "Sharpe_dir"), ("pnl_net", "Sharpe_dir_net")]:
        s = daily[col]
        m[name] = float(s.mean() / (s.std() + 1e-12) * np.sqrt(252))
    # (b) cross-sectional long/short: long top-20%, short bottom-20% forecasts within region
    def ls(g):
        if len(g) < 5:
            return np.nan
        q = g.p.rank(pct=True)
        return g.y[q > 0.8].mean() - g.y[q <= 0.2].mean()
    lsr = df.groupby(["d", "g"]).apply(ls, include_groups=False).groupby(level=0).mean().dropna()
    m["Sharpe_LS"] = float(lsr.mean() / (lsr.std() + 1e-12) * np.sqrt(252)) if len(lsr) > 10 else 0.0
    # relative (market-neutral) accuracy: does the model know which assets beat their regional peers today?
    df["y_xs"] = df.y - df.groupby(["g", "d"]).y.transform("mean")
    df["p_xs"] = df.p - df.groupby(["g", "d"]).p.transform("mean")
    nz = df.y_xs != 0
    m["DA_xs_%"] = float(np.mean(np.sign(df.p_xs[nz]) == np.sign(df.y_xs[nz])) * 100)
    return m


def evaluate(pred_norm, ends, panel):
    """pred_norm: (n, H) model outputs in training-target space for window ends `ends`."""
    P = panel.np_
    tid = P["tid"][ends]
    close_t = P["close"][ends].astype(np.float64)
    y_ret = P["y_ret"][ends].astype(np.float64)
    sigma = P["sigma"][ends].astype(np.float64)
    r_pred = pred_norm.astype(np.float64) * sigma[:, None]
    regions = np.array(panel.meta["region"])[tid]
    dates = P["dates"][ends]
    out = {}
    # ---------- 1-day horizon
    c_true = P["y_close"][ends, 0].astype(np.float64)
    c_pred = close_t * np.exp(r_pred[:, 0])
    out.update(price_metrics(close_t, c_true, c_pred))
    out.update(return_metrics(y_ret[:, 0], r_pred[:, 0], dates, tid, regions))
    # ---------- 5-day horizon (cumulative)
    H = y_ret.shape[1]
    r5, p5 = y_ret.sum(1), r_pred.sum(1)
    out[f"R2_oos_{H}d_%"] = float((1 - np.sum((r5 - p5) ** 2) / np.sum(r5 ** 2)) * 100)
    out[f"DA_{H}d_%"] = float(np.mean(np.sign(p5[r5 != 0]) == np.sign(r5[r5 != 0])) * 100)
    out[f"MAPE_{H}d_%"] = float(np.mean(np.abs(np.exp(p5) - np.exp(r5)) / np.exp(r5)) * 100)
    return out, r_pred


def naive_metrics(ends, panel):
    """Random-walk baseline: tomorrow's price = today's price (zero predicted return)."""
    zeros = np.zeros((len(ends), panel.horizon))
    out, _ = evaluate(zeros, ends, panel)
    out["DA_%"] = float("nan")  # undefined for a zero forecast
    out[f"DA_{panel.horizon}d_%"] = float("nan")
    for k in ("IC", "IC_IR", "Sharpe_dir", "Sharpe_dir_net", "Sharpe_LS", "corr"):
        out[k] = 0.0
    return out


def breakdown(r_pred1, ends, panel, by: str):
    """Per-group metrics (by = 'vol_tier' | 'asset_class' | 'ticker')."""
    P = panel.np_
    tid = P["tid"][ends]
    if by == "ticker":
        grp = np.array(panel.meta["tickers"])[tid]
    else:
        grp = np.array(panel.meta[by])[tid]
    y = P["y_ret"][ends, 0]
    close_t = P["close"][ends]
    c_true = P["y_close"][ends, 0]
    rows = {}
    for g in np.unique(grp):
        m = grp == g
        nz = m & (y != 0)
        cp = close_t[m] * np.exp(r_pred1[m])
        rows[g] = {
            "n": int(m.sum()),
            "DA_%": float(np.mean(np.sign(r_pred1[nz]) == np.sign(y[nz])) * 100),
            "R2_oos_%": float((1 - np.sum((y[m] - r_pred1[m]) ** 2) / np.sum(y[m] ** 2)) * 100),
            "MAPE_%": float(np.mean(np.abs(cp - c_true[m]) / c_true[m]) * 100),
            "MAPE_naive_%": float(np.mean(np.abs(close_t[m] - c_true[m]) / c_true[m]) * 100),
        }
    return rows


# ----------------------------------------------------------------------------- forecast cone
CONE_LEVELS = (80, 95)


def cone_quantiles(y_ret, sigma, levels=CONE_LEVELS):
    """Spread of realised cumulative k-day log returns, in units of the volatility known at t.

    Computed on the VALIDATION period only, centred on the median, so it can be added to any
    model's predicted path to form a calibrated prediction band:
        band_k(t) = pred_cum_k(t) + q_k * sigma(t)
    Returns {level: (lo[H], hi[H])}."""
    z = np.cumsum(y_ret, 1) / sigma[:, None]
    med = np.median(z, 0)
    out = {}
    for L in levels:
        a = (100 - L) / 2
        out[L] = (np.percentile(z, a, 0) - med, np.percentile(z, 100 - a, 0) - med)
    return out


def cone_coverage(pred_ret, y_ret, sigma, q):
    """Share of (test window, horizon step) pairs whose realised cumulative return lies inside the band."""
    pc, yc = np.cumsum(pred_ret, 1), np.cumsum(y_ret, 1)
    ok = np.isfinite(pc).all(1)
    res = {}
    for L, (lo, hi) in q.items():
        inside = (yc[ok] >= pc[ok] + lo * sigma[ok, None]) & (yc[ok] <= pc[ok] + hi * sigma[ok, None])
        res[L] = {"all": float(inside.mean() * 100), "by_step": (inside.mean(0) * 100).tolist(),
                  "width_5d_%": float(np.median((hi[-1] - lo[-1]) * sigma[ok]) * 100)}
    return res
