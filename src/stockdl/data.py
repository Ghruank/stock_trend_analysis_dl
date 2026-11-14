"""GPU-friendly sliding-window sampler over the pre-engineered panel (data/dataset.npz).

The whole feature matrix lives on the device; a batch of windows is gathered with one
index operation, which is far faster than a python DataLoader for this tabular problem.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Panel:
    X: torch.Tensor          # (N, F)   scaled features for the chosen feature set
    marks: torch.Tensor      # (N, 3)   month, day, weekday (int)
    asset: torch.Tensor      # (N,)     asset id
    target: torch.Tensor     # (N, H)   training target (normalised)
    ends: dict               # split -> np.ndarray of window end rows
    np_: dict                # numpy copies needed for evaluation
    meta: dict
    feature_set: str
    lookback: int
    horizon: int


def load_panel(path: str, feature_set: str = "full", lookback: int = 96, device: str = "cpu",
               target_clip: float = 5.0, target: str = "ret") -> Panel:
    """target = "ret": volatility-normalised future returns.
       target = "xs" : same, but cross-sectionally demeaned within (region, date) -> market-neutral,
                       i.e. the model learns which assets will beat their peers, not where the market goes."""
    z = np.load(path)
    meta = json.load(open(os.path.join(os.path.dirname(path), "dataset_meta.json")))
    H = int(meta["horizon"])
    if feature_set == "full":
        X = z["X_full"]
    elif feature_set == "base":
        X = z["X_full"][:, z["base_idx"]]
    elif feature_set.startswith("drop:"):           # ablation: full minus a feature group
        groups = feature_set[5:].split("+")
        names = meta["features_full"]
        keep = [i for i, n in enumerate(names) if feature_group(n) not in groups]
        X = z["X_full"][:, keep]
    else:
        raise ValueError(feature_set)
    tid, split, valid = z["ticker_id"], z["split"], z["valid"]
    sigma, close = z["sigma"], z["close"]
    # volatility-normalised future log returns (stationary, comparable across assets)
    tgt = np.clip(z["y_ret"] / sigma[:, None], -target_clip, target_clip)
    if target == "xs":
        import pandas as pd
        reg = np.array(meta["region"])[tid]
        df = pd.DataFrame(np.where(valid[:, None], tgt, np.nan))
        key = [reg, z["dates"]]
        tgt = (df - df.groupby(key).transform("mean")).values
    tgt = np.nan_to_num(tgt).astype(np.float32)

    # window end i is usable if the whole window [i-L+1, i] belongs to the same asset
    # and the row's targets exist.  Future marks (i+1..i+H) always exist when targets do.
    N = len(tid)
    ok = valid.copy()
    start = np.arange(N) - lookback + 1
    ok &= start >= 0
    ok[ok] &= tid[start[ok]] == tid[ok]
    ends = {s: np.where(ok & (split == k))[0] for k, s in enumerate(["train", "val", "test"])}
    dev = torch.device(device)
    return Panel(
        X=torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev),
        marks=torch.from_numpy(z["marks"].astype(np.int64)).to(dev),
        asset=torch.from_numpy(tid.astype(np.int64)).to(dev),
        target=torch.from_numpy(tgt).to(dev),
        ends=ends,
        np_={"y_ret": z["y_ret"], "y_close": z["y_close"], "sigma": sigma, "close": close, "tid": tid,
             "dates": z["dates"]},
        meta=meta, feature_set=feature_set, lookback=lookback, horizon=H,
    )


def feature_group(name: str) -> str:
    """Map a feature name to a semantic group (used for ablations / documentation)."""
    if name.startswith("B_"):
        return "base"
    if name.startswith("M_"):
        return "market"
    if name.startswith("X_"):
        return "cross_sectional"
    if name.startswith("C_"):
        return "calendar"
    n = name[2:]
    if n.startswith("ret"):
        return "momentum"
    if n in ("gap", "intraday", "hl_range", "clv", "upper_shadow", "lower_shadow"):
        return "candle"
    if n.startswith(("vol", "parkinson", "garman", "atr", "skew", "kurt", "autocorr")) and not n.startswith("vol_ratio_5"):
        return "volatility"
    if n.startswith(("rsi", "macd", "stoch", "cci", "mfi", "adx", "di_")):
        return "oscillator"
    if n.startswith(("px_sma", "bb_", "dist_52w", "fracdiff")):
        return "trend"
    if n.startswith(("vol_ratio_5", "dlogvol", "obv", "amihud", "turnover")):
        return "volume_liquidity"
    if n.startswith(("beta", "resid", "rel_strength", "corr_mkt")):
        return "market"
    return "other"


def gather(p: Panel, ends: np.ndarray, label_len: int = 48):
    """Build one batch for window ends `ends` (numpy int array)."""
    dev = p.X.device
    e = torch.as_tensor(ends, device=dev)
    L, H = p.lookback, p.horizon
    win = e[:, None] + torch.arange(-L + 1, 1, device=dev)[None, :]          # (B, L)
    fut = e[:, None] + torch.arange(1, H + 1, device=dev)[None, :]           # (B, H)
    x = p.X[win]                                                             # (B, L, F)
    mark_enc = p.marks[win]
    mark_dec = torch.cat([mark_enc[:, -label_len:], p.marks[fut]], 1)         # (B, label+H, 3)
    return {"x": x, "mark_enc": mark_enc, "mark_dec": mark_dec, "asset": p.asset[e], "y": p.target[e]}


def iterate(p: Panel, split: str, batch_size: int, shuffle: bool, rng: np.random.Generator | None = None,
            label_len: int = 48):
    ends = p.ends[split]
    if shuffle:
        ends = ends[rng.permutation(len(ends))]
    for i in range(0, len(ends), batch_size):
        yield gather(p, ends[i:i + batch_size], label_len)
