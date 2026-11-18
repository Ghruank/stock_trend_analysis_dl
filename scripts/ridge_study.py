"""Linear reference model + feature-family ablation (CPU, seconds).

* ridge_xs   : ridge regression on the market-neutral (cross-sectional) target
* ablation   : drop one feature family at a time from the 87 features, refit ridge (alpha re-selected
               on validation), report validation correlation and test IC / L-S Sharpe.
Appends rows to results/baselines/runs.jsonl.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
from stockdl.data import feature_group, load_panel  # noqa: E402
from stockdl.metrics import breakdown, evaluate  # noqa: E402

OUT = os.path.join(ROOT, "results", "baselines")
ALPHAS = [10, 100, 1000, 10000, 100000]


def fit_ridge(p, cols=None):
    X = p.X.numpy() if cols is None else p.X.numpy()[:, cols]
    Y = p.target.numpy()
    tr, va, te = p.ends["train"], p.ends["val"], p.ends["test"]
    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(X[tr], Y[tr])
        pv = m.predict(X[va])
        c = np.corrcoef(pv[:, 0], Y[va, 0])[0, 1]
        if best is None or c > best[0]:
            best = (c, a, m)
    c, a, m = best
    return m.predict(X[te]), a, float(c)


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    have = {r["key"] for r in rows}
    f = open(os.path.join(OUT, "runs.jsonl"), "a")
    p = load_panel(os.path.join(ROOT, "data", "dataset.npz"), "full", 96, "cpu", target="xs")
    te = p.ends["test"]
    if "baseline:ridge_xs" not in have:
        pred, a, c = fit_ridge(p)
        met, rp = evaluate(pred, te, p)
        np.savez_compressed(os.path.join(OUT, "preds", "ridge_xs_full.npz"), pred=pred.astype(np.float32), ends=te)
        f.write(json.dumps({"key": "baseline:ridge_xs", "phase": "baseline", "model": "ridge_xs", "features": "full",
                            "alpha": a, "val_corr": c, "test": met, "by_vol_tier": breakdown(rp[:, 0], te, p, "vol_tier"),
                            "by_asset_class": breakdown(rp[:, 0], te, p, "asset_class")}) + "\n")
        f.flush()
        print("ridge_xs", a, round(c, 4), {k: round(v, 4) for k, v in met.items() if k in ("DA_%", "IC", "Sharpe_LS", "DA_xs_%")})
    names = p.meta["features_full"]
    groups = sorted({feature_group(n) for n in names})
    for g in ["(none)"] + groups:
        key = f"ridge_ablation:{g}"
        if key in have:
            continue
        cols = [i for i, n in enumerate(names) if feature_group(n) != g]
        pred, a, c = fit_ridge(p, cols)
        met, _ = evaluate(pred, te, p)
        f.write(json.dumps({"key": key, "phase": "ridge_ablation", "model": "ridge_xs", "features": f"drop:{g}",
                            "n_dropped": len(names) - len(cols), "alpha": a, "val_corr": c, "test": met}) + "\n")
        f.flush()
        print(f"drop {g:18s} n={len(names) - len(cols):2d} valcorr={c:.4f} IC={met['IC']:.4f} LS={met['Sharpe_LS']:.2f}")
    # only-base reference (8 features) for the linear model
    if "ridge_ablation:only_base" not in have:
        cols = [i for i, n in enumerate(names) if feature_group(n) == "base"]
        pred, a, c = fit_ridge(p, cols)
        met, _ = evaluate(pred, te, p)
        f.write(json.dumps({"key": "ridge_ablation:only_base", "phase": "ridge_ablation", "model": "ridge_xs",
                            "features": "only:base", "n_dropped": len(names) - len(cols), "alpha": a, "val_corr": c,
                            "test": met}) + "\n")
        print(f"only base           valcorr={c:.4f} IC={met['IC']:.4f} LS={met['Sharpe_LS']:.2f}")


if __name__ == "__main__":
    main()
