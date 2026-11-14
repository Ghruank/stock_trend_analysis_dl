"""Feature engineering for daily stock forecasting.

Every feature at row t uses ONLY information available at the close of day t
(no look-ahead). Targets are next-day quantities (t+1 ... t+H).

Two feature sets are produced:
  * BASE    - 8 basic price/volume features made stationary: return, open/high/low
              relative to close, volume z-score, ROC, RSI and Bollinger width (period 5).
  * FULL    - BASE + ~60 additional engineered features (volatility estimators,
              oscillators, trend, volume/liquidity, market context, cross-sectional
              ranks, fractional differencing, calendar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------------- helpers
def rsi_simple(close: pd.Series, n: int) -> pd.Series:
    """RSI with simple moving averages."""
    d = close.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + EPS))


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + EPS))


def frac_diff_weights(d: float, thresh: float = 1e-4, max_len: int = 500) -> np.ndarray:
    """Fixed-window fractional differencing weights (Lopez de Prado, AFML ch.5)."""
    w = [1.0]
    k = 1
    while k < max_len:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float = 0.4) -> pd.Series:
    w = frac_diff_weights(d)
    return series.rolling(len(w)).apply(lambda x: float(np.dot(w, x)), raw=True)


def zscore_roll(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / (s.rolling(n).std() + EPS)


# ----------------------------------------------------------------------------- per ticker
def ticker_features(df: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    """df: one ticker, indexed by date, columns open/high/low/close/volume.
    mkt: market context aligned on the same dates (already lagged where needed)."""
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    f = pd.DataFrame(index=df.index)
    lc = np.log(c)
    r = lc.diff()  # daily log return

    sma5, sd5 = c.rolling(5).mean(), c.rolling(5).std()

    # ---------------- BASE (basic price / volume information, stationary) ----------------
    f["B_ret1"] = r
    f["B_open_rel"] = np.log(o / c)
    f["B_high_rel"] = np.log(h / c)
    f["B_low_rel"] = np.log(l / c)
    f["B_logvol_z"] = zscore_roll(np.log1p(v), 60)
    f["B_roc5"] = c.pct_change(5)
    f["B_rsi5"] = rsi_simple(c, 5) / 100 - 0.5
    f["B_bbw5"] = 4 * sd5 / (sma5 + EPS)

    # ---------------- FULL (new engineered features) ----------------
    # multi-horizon momentum
    for n in (2, 3, 5, 10, 20, 60, 120, 250):
        f[f"F_ret{n}"] = lc.diff(n)
    # candle anatomy
    f["F_gap"] = np.log(o / c.shift(1))
    f["F_intraday"] = np.log(c / o)
    f["F_hl_range"] = np.log(h / l)
    f["F_clv"] = ((c - l) - (h - c)) / (h - l + EPS)
    f["F_upper_shadow"] = np.log(h / np.maximum(o, c))
    f["F_lower_shadow"] = np.log(np.minimum(o, c) / l)
    # volatility estimators
    for n in (5, 20, 60):
        f[f"F_vol{n}"] = r.rolling(n).std()
    hl2 = np.log(h / l) ** 2
    f["F_parkinson20"] = np.sqrt(hl2.rolling(20).mean() / (4 * np.log(2)))
    gk = 0.5 * hl2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    f["F_garman_klass20"] = np.sqrt(gk.rolling(20).mean().clip(lower=0))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["F_atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    f["F_vol_ratio"] = f["F_vol5"] / (f["F_vol60"] + EPS)
    f["F_vol_of_vol"] = f["F_vol20"].rolling(60).std() / (f["F_vol20"].rolling(60).mean() + EPS)
    f["F_skew60"] = r.rolling(60).skew()
    f["F_kurt60"] = r.rolling(60).kurt()
    f["F_autocorr20"] = r.rolling(60).apply(lambda x: pd.Series(x).autocorr(1), raw=True)
    # oscillators
    f["F_rsi14"] = rsi_wilder(c, 14) / 100 - 0.5
    f["F_rsi2"] = rsi_wilder(c, 2) / 100 - 0.5
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / c
    sig = macd.ewm(span=9, adjust=False).mean()
    f["F_macd"], f["F_macd_hist"] = macd, macd - sig
    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    k = (c - ll14) / (hh14 - ll14 + EPS)
    f["F_stoch_k"] = k - 0.5
    f["F_stoch_d"] = k.rolling(3).mean() - 0.5
    tp = (h + l + c) / 3
    f["F_cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True) + EPS) / 100
    mf = tp * v
    pos = mf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(14).sum()
    f["F_mfi14"] = (100 - 100 / (1 + pos / (neg + EPS))) / 100 - 0.5
    # ADX
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + EPS)
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / (atr + EPS)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + EPS)
    f["F_adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean() / 100
    f["F_di_diff"] = (pdi - ndi) / 100
    # trend / mean reversion
    for n in (10, 20, 50, 200):
        f[f"F_px_sma{n}"] = np.log(c / c.rolling(n).mean())
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["F_bb_pctb20"] = (c - (sma20 - 2 * sd20)) / (4 * sd20 + EPS) - 0.5
    f["F_bb_width20"] = 4 * sd20 / sma20
    f["F_dist_52w_high"] = np.log(c / h.rolling(250).max())
    f["F_dist_52w_low"] = np.log(c / l.rolling(250).min())
    f["F_fracdiff_logpx"] = zscore_roll(frac_diff(lc, 0.4), 250)
    # volume / liquidity
    lv = np.log1p(v)
    f["F_vol_ratio_5_20"] = lv.rolling(5).mean() - lv.rolling(20).mean()
    f["F_dlogvol"] = lv.diff()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["F_obv_slope20"] = (obv - obv.shift(20)) / (v.rolling(20).sum() + EPS)
    f["F_amihud20"] = np.log((r.abs() / (c * v + EPS)).rolling(20).mean() + 1e-15)
    f["F_turnover_z"] = zscore_roll(np.log(c * v + 1), 250)
    # market context (stock vs NIFTY)
    rm = mkt["home_ret"]
    cov = r.rolling(60).cov(rm)
    beta = cov / (rm.rolling(60).var() + EPS)
    f["F_beta60"] = beta
    f["F_resid_ret1"] = r - beta * rm
    f["F_resid_ret5"] = f["F_resid_ret1"].rolling(5).sum()
    f["F_rel_strength20"] = lc.diff(20) - mkt["home_ret"].rolling(20).sum()
    f["F_corr_mkt60"] = r.rolling(60).corr(rm)
    for col in mkt.columns:
        f[f"M_{col}"] = mkt[col]
    # calendar (also used as Informer time marks)
    idx = df.index
    f["C_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    f["C_dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    f["C_month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    f["C_month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    f["C_dom"] = (idx.day - 1) / 30 - 0.5
    return f


def market_context(market: pd.DataFrame, dates: pd.DatetimeIndex, region: str) -> pd.DataFrame:
    """Market features aligned to the asset's own trading dates.

    IN assets (NSE, close 15:30 IST): home index = NIFTY 50, fear gauge = India VIX.
        S&P500, crude, USD/INR close AFTER the NSE close -> lagged by 1 day.
    US assets (NYSE / COMEX / NYMEX): home index = S&P 500, fear gauge = CBOE VIX.
        NIFTY closed earlier the same day -> same-day value is legitimately known.
        Crude and the dollar index are lagged 1 day to be safe.
    """
    m = market.copy().sort_index()
    m = m.reindex(m.index.union(dates)).ffill(limit=5)
    out = pd.DataFrame(index=m.index)
    home, vix = ("nifty", "vix") if region == "IN" else ("sp500", "vix_us")
    hl = np.log(m[home])
    out["home_ret"] = hl.diff()
    out["home_ret5"] = hl.diff(5)
    out["home_ret20"] = hl.diff(20)
    out["home_vol20"] = out["home_ret"].rolling(20).std()
    out["home_px_sma50"] = np.log(m[home] / m[home].rolling(50).mean())
    out["vix_log"] = np.log(m[vix]) - 3.0
    out["vix_chg5"] = np.log(m[vix]).diff(5)
    ext = {"IN": [("sp500", 1), ("crude", 1), ("usdinr", 1)],
           "US": [("nifty", 0), ("crude", 1), ("dxy", 1)]}[region]
    for j, (k, lag) in enumerate(ext):
        s = np.log(m[k].where(m[k] > 0))
        out[f"ext{j}_ret1"] = s.diff().shift(lag)
        out[f"ext{j}_ret5"] = s.diff(5).shift(lag)
    out = out.reindex(dates)
    return out


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks among assets of the same region on the same day."""
    panel = panel.copy()
    for col, name in [("F_ret5", "X_rank_ret5"), ("F_ret20", "X_rank_ret20"),
                      ("F_ret60", "X_rank_ret60"), ("F_vol20", "X_rank_vol20"),
                      ("B_logvol_z", "X_rank_volume"), ("B_ret1", "X_rank_ret1")]:
        panel[name] = panel.groupby(["region", "date"])[col].rank(pct=True) - 0.5
    panel["X_ret1_demeaned"] = panel["B_ret1"] - panel.groupby(["region", "date"])["B_ret1"].transform("mean")
    return panel


def build_panel(prices: pd.DataFrame, market: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = clean_prices(prices)
    market = market.copy()
    market.index = pd.to_datetime(market.index)
    out = []
    for t, d in prices.groupby("ticker"):
        region, aclass = d["region"].iloc[0], d["asset_class"].iloc[0]
        d = d.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]]
        mkt = market_context(market, d.index, region)
        f = ticker_features(d, mkt)
        lc = np.log(d["close"])
        r = lc.diff()
        # volatility used to normalise targets: EWMA std of returns, known at t
        sigma = r.ewm(span=20, min_periods=20).std()
        f["sigma"] = sigma
        f["close"] = d["close"]
        for k in range(1, horizon + 1):
            f[f"y_ret{k}"] = r.shift(-k)                   # raw next-k-th-day log return
            f[f"y_close{k}"] = d["close"].shift(-k)        # raw future close (for price metrics)
        f["ticker"], f["region"], f["asset_class"] = t, region, aclass
        out.append(f.copy().reset_index())
    panel = pd.concat(out, ignore_index=True)
    panel = add_cross_sectional(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def feature_columns(panel: pd.DataFrame) -> dict:
    base = [c for c in panel.columns if c.startswith("B_")]
    full = base + [c for c in panel.columns if c[:2] in ("F_", "M_", "X_", "C_")]
    return {"base": base, "full": full}


# Verified one-off gaps that are NOT economic returns (unadjusted by Yahoo):
#   ADANIENT 2015-06-03 : demerger of Adani Ports/Power undertakings
#   VEDL     2026-04-30 : Vedanta group demerger (ex-date)
#   NATGAS   2026-01-29 : continuous front-month futures roll gap
#   GOLDBEES 2019-12-19/23 : unit split (1:100) with a bad interim print
# YESBANK 2020-03-06 (-82%) is a REAL crash (RBI moratorium) and is intentionally kept.
CORPORATE_ACTIONS = [("ADANIENT", "2015-06-03"), ("VEDL", "2026-04-30"), ("NATGAS", "2026-01-29"),
                     ("GOLDBEES", "2019-12-19"), ("GOLDBEES", "2019-12-23")]


def clean_prices(prices: pd.DataFrame, spike: float = 0.35, verbose: bool = True) -> pd.DataFrame:
    """Data-quality fixes for the Yahoo snapshot.

    1. Single-day spikes: a |log-return| > `spike` that is immediately reversed
       (opposite sign, similar size) is a bad tick -> the row is dropped.
    2. Listed corporate actions / roll gaps (CORPORATE_ACTIONS) -> all earlier OHLC values
       are back-adjusted by the gap ratio (standard price-adjustment practice).
    """
    out = []
    for t, d in prices.groupby("ticker"):
        d = d.sort_values("date").reset_index(drop=True)
        r = np.log(d["close"]).diff()
        bad = (r.abs() > spike) & (r.shift(-1).abs() > spike) & (np.sign(r) != np.sign(r.shift(-1))) \
            & ((r + r.shift(-1)).abs() < 0.15)
        if bad.any():
            if verbose:
                print(f"[clean] {t}: dropping bad ticks {d.loc[bad, 'date'].dt.date.tolist()}")
            d = d[~bad].reset_index(drop=True)
        for tk, day in CORPORATE_ACTIONS:
            if tk != t:
                continue
            i = d.index[d["date"] == pd.Timestamp(day)]
            if len(i) == 0:
                continue
            i = i[0]
            ratio = d.loc[i, "close"] / d.loc[i - 1, "close"]
            if verbose:
                print(f"[clean] {t}: back-adjusting {day} ratio={ratio:.3f}")
            for k in ("open", "high", "low", "close"):
                d.loc[: i - 1, k] *= ratio
        out.append(d)
    return pd.concat(out, ignore_index=True)
