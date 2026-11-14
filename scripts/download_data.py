"""Download a frozen daily OHLCV snapshot for the stock universe + market context series.

Usage: python scripts/download_data.py --end 2026-11-20
Writes data/prices.csv (long format: date,ticker,asset_class,region,open,high,low,close,volume)
       data/market.csv (wide: date, nifty, vix, usdinr, sp500, crude, vix_us, dxy)
"""
import argparse, os
import pandas as pd
import yfinance as yf

STOCKS = ["TCS", "INFY", "WIPRO", "HCLTECH", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN",
          "AXISBANK", "KOTAKBANK", "HINDUNILVR", "ITC", "NESTLEIND", "ASIANPAINT", "BAJFINANCE",
          "LT", "SUNPHARMA", "MARUTI", "TATASTEEL", "ADANIENT", "ADANIPORTS", "BHARTIARTL",
          "M&M", "TITAN", "ONGC"]
# high-volatility / mid-cap Indian names (blog: "volatile" stocks such as Adani)
STOCKS_IN_VOLATILE = ["YESBANK", "SUZLON", "IDEA", "VEDL", "TATAPOWER", "ZEEL", "BHEL", "SAIL", "IDFCFIRSTB", "JINDALSTEL"]
STOCKS_US = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "KO", "AMZN", "XOM"]
COMMODITIES = {"GC=F": "GOLD", "SI=F": "SILVER", "CL=F": "CRUDE", "NG=F": "NATGAS", "HG=F": "COPPER",
               "GOLDBEES.NS": "GOLDBEES"}
MARKET = {"^NSEI": "nifty", "^INDIAVIX": "vix", "INR=X": "usdinr", "^GSPC": "sp500", "CL=F": "crude",
          "^VIX": "vix_us", "DX-Y.NYB": "dxy"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2026-11-20")  # exclusive in yfinance
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = {}  # yahoo symbol -> (name, asset_class, region)
    for s_ in STOCKS: spec[s_ + ".NS"] = (s_, "equity_in", "IN")
    for s_ in STOCKS_IN_VOLATILE: spec[s_ + ".NS"] = (s_, "equity_in", "IN")
    for s_ in STOCKS_US: spec[s_] = (s_, "equity_us", "US")
    for k, v in COMMODITIES.items(): spec[k] = (v, "commodity", "IN" if k.endswith(".NS") else "US")
    tick = list(spec)
    raw = yf.download(tick, start=a.start, end=a.end, auto_adjust=True, progress=False, group_by="ticker")
    rows = []
    for t in tick:
        d = raw[t].dropna(subset=["Close"]).copy()
        d = d[(d["Volume"] > 0) & (d["Close"] > 0) & (d["Low"] > 0)]
        d.columns = [c.lower() for c in d.columns]
        d["ticker"], d["asset_class"], d["region"] = spec[t]
        d.index.name = "date"
        rows.append(d.reset_index()[["date", "ticker", "asset_class", "region", "open", "high", "low", "close", "volume"]])
    prices = pd.concat(rows)
    prices.to_csv(os.path.join(a.out, "prices.csv"), index=False, float_format="%.4f")
    m = yf.download(list(MARKET), start=a.start, end=a.end, auto_adjust=True, progress=False)["Close"]
    m = m.rename(columns=MARKET)
    m.index.name = "date"
    m.to_csv(os.path.join(a.out, "market.csv"), float_format="%.5f")
    print(prices.groupby("ticker").date.agg(["min", "max", "count"]))
    print(m.notna().sum())

if __name__ == "__main__":
    main()
