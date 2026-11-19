"""Collect every experiment artefact into compact JSON for the website / slides / report.

python scripts/export_web_data.py --results results --out website/public/data
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.data import feature_group, load_panel  # noqa: E402
from stockdl.metrics import breakdown, cone_coverage, cone_quantiles, evaluate  # noqa: E402

MODELS = ["lstm", "transformer", "informer"]
KEY = ["DA_xs_%", "DA_%", "R2_oos_%", "IC", "IC_IR", "MAPE_%", "RMSE", "MAE", "R2_price", "Sharpe_dir", "Sharpe_dir_net",
       "Sharpe_LS", "corr", "DA_5d_%", "R2_oos_5d_%", "MAPE_5d_%"]


def rnd(x, k=5):
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, k)
    if isinstance(x, dict):
        return {a: rnd(b, k) for a, b in x.items()}
    if isinstance(x, list):
        return [rnd(b, k) for b in x]
    return x


def align(z, te):
    """Place saved predictions onto the full test index (NaN where missing)."""
    out = np.full((len(te), z["pred"].shape[1]), np.nan)
    pos = np.searchsorted(te, z["ends"])
    out[pos] = z["pred"]
    return out


def lst(a, k=4):
    return [None if not np.isfinite(x) else round(float(x), k) for x in a]


def load_rows(results):
    rows = []
    for f in glob.glob(os.path.join(results, "*", "runs.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "website", "public", "data"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "series"), exist_ok=True)
    rows = load_rows(a.results)
    meta = json.load(open(os.path.join(ROOT, "data", "dataset_meta.json")))
    p = load_panel(os.path.join(ROOT, "data", "dataset.npz"), "full", 96, "cpu")
    te = p.ends["test"]
    P = p.np_
    tick = meta["tickers"]

    out = {"meta": {k: meta[k] for k in ("tickers", "asset_class", "region", "vol_tier", "train_ann_vol", "splits",
                                          "rows_valid_per_split", "horizon")}}
    out["meta"]["features"] = [{"name": n, "group": feature_group(n)} for n in meta["features_full"]]
    out["meta"]["n_features"] = {"base": len(meta["features_base"]),
                                 "full": len(meta["features_full"])}
    out["baselines"] = {r["model"]: rnd({k: r["test"].get(k) for k in KEY}) for r in rows if r.get("phase") == "baseline"}
    out["baseline_breakdown"] = {r["model"]: rnd({"vol_tier": r.get("by_vol_tier"), "asset_class": r.get("by_asset_class")})
                                 for r in rows if r.get("phase") == "baseline" and "by_vol_tier" in r}

    # ---------------- search
    out["search"] = [rnd({"model": r["model"], "cfg": r["cfg"], "val_loss": r["best_val_loss"], "epochs": r["epochs_run"],
                          "params": r["n_params"], "minutes": r["train_minutes"],
                          "val": {k: r["val"].get(k) for k in ("DA_%", "R2_oos_%", "IC")},
                          "history": r["history"]}) for r in rows if r.get("phase") == "search"]

    # ---------------- final (per seed) + summary
    for r in rows:
        if r.get("phase") == "xs":
            r["phase"], r["features"] = "final", "xs_" + r["features"]
        if r.get("phase") == "ensemble_xs":
            r["phase"], r["features"] = "ensemble", "xs_" + r["features"]
    # backfill metrics added after the early runs finished (DA_xs_%) from the saved predictions
    for r in rows:
        if r.get("key") == "baseline:ridge" and r["test"].get("DA_xs_%") is None:
            f = os.path.join(a.results, "baselines", "preds", "ridge_full.npz")
            if os.path.exists(f):
                te = p.ends["test"]
                r["test"]["DA_xs_%"] = evaluate(align(np.load(f), te), te, p)[0]["DA_xs_%"]
                out["baselines"]["ridge"]["DA_xs_%"] = round(r["test"]["DA_xs_%"], 5)
        if r.get("phase") in ("final", "ensemble") and "test" in r and r["test"].get("DA_xs_%") is None and r.get("model") in MODELS:
            fs = r["features"]
            name = (f"final_{r['model']}_{fs}_s{r['seed']}.npz" if r["phase"] == "final" else f"ensemble_{r['model']}_{fs}.npz")
            if fs.startswith("xs_"):
                name = name.replace("final_", "xs_").replace(f"_{fs}", f"_{fs[3:]}").replace("ensemble_", "ensemble_xs_")
            f = os.path.join(a.results, r["model"], "preds", name)
            if os.path.exists(f):
                te = p.ends["test"]
                met, _ = evaluate(align(np.load(f), te), te, p)
                r["test"]["DA_xs_%"] = met["DA_xs_%"]
    finals = [r for r in rows if r.get("phase") == "final"]
    out["runs"] = [rnd({"model": r["model"], "features": r["features"], "seed": r["seed"], "cfg": r["cfg"],
                        "val_loss": r["best_val_loss"], "epochs": r["epochs_run"], "params": r["n_params"],
                        "minutes": r["train_minutes"], "test": {k: r["test"].get(k) for k in KEY},
                        "history": r["history"]}) for r in finals]
    summ = []
    df = pd.DataFrame([{"model": r["model"], "features": r["features"], **{k: r["test"].get(k) for k in KEY}} for r in finals]).astype({k: float for k in KEY}, errors="ignore")
    for (m, fs), g in df.groupby(["model", "features"]):
        ens = next((r for r in rows if r.get("phase") == "ensemble" and r["model"] == m and r["features"] == fs), None)
        summ.append(rnd({"model": m, "features": fs, "n_seeds": len(g),
                         "mean": {k: float(g[k].mean()) for k in KEY},
                         "std": {k: float(g[k].std()) if len(g) > 1 else 0.0 for k in KEY},
                         "ensemble": {k: ens["test"].get(k) for k in KEY} if ens else None}))
    out["summary"] = summ
    out["breakdown"] = {f"{r['model']}:{r['features']}": rnd({"vol_tier": r["by_vol_tier"], "asset_class": r["by_asset_class"],
                                                              "ticker": r.get("by_ticker")})
                        for r in rows if r.get("phase") == "ensemble"}

    # ---------------- ablation
    abl = [r for r in rows if r.get("phase") == "ablation_xs"] or [r for r in rows if r.get("phase") == "ablation"]
    if abl:
        m = abl[0]["model"]
        ref_fs = "xs_full" if abl[0]["phase"] == "ablation_xs" else "full"
        ref = [r for r in finals if r["model"] == m and r["features"] == ref_fs and r["seed"] in (0, 1)]
        ref_v = np.mean([r["best_val_loss"] for r in ref])
        ref_t = {k: np.mean([r["test"][k] for r in ref]) for k in ("DA_%", "IC", "R2_oos_%", "Sharpe_LS")}
        groups = sorted({r["features"].split(":")[1] for r in abl})
        out["ablation"] = {"model": m, "target": ref_fs, "ref_val_loss": rnd(float(ref_v)), "ref_test": rnd(ref_t), "groups": []}
        for gname in groups:
            rr = [r for r in abl if r["features"] == f"drop:{gname}"]
            if not rr:
                continue
            out["ablation"]["groups"].append(rnd({
                "group": gname, "n_features_dropped": sum(feature_group(n) == gname for n in meta["features_full"]),
                "val_loss": float(np.mean([r["best_val_loss"] for r in rr])),
                "delta_val_loss": float(np.mean([r["best_val_loss"] for r in rr]) - ref_v),
                "test": {k: float(np.mean([r["test"][k] for r in rr])) for k in ("DA_%", "IC", "R2_oos_%", "Sharpe_LS")}}))

    # ---------------- ridge feature-family ablation (linear model, market-neutral target)
    ra = [r for r in rows if r.get("phase") == "ridge_ablation"]
    if ra:
        ref = next(r for r in ra if r["features"] == "drop:(none)")
        out["ridge_ablation"] = rnd({"ref": {"val_corr": ref["val_corr"], **{k: ref["test"][k] for k in ("IC", "Sharpe_LS", "DA_xs_%")}},
                                     "groups": [{"group": r["features"].split(":")[1], "n": r["n_dropped"], "val_corr": r["val_corr"],
                                                 "IC": r["test"]["IC"], "Sharpe_LS": r["test"]["Sharpe_LS"], "DA_xs_%": r["test"].get("DA_xs_%")}
                                                for r in ra if r["features"] != "drop:(none)"]})

    # ---------------- per-ticker prediction series + equity curves
    dates = pd.to_datetime(P["dates"][te], unit="D")
    tid = P["tid"][te]
    close_t, c_next = P["close"][te].astype(float), P["y_close"][te, 0].astype(float)
    y = P["y_ret"][te, 0].astype(float)
    preds, preds5 = {}, {}
    for m in MODELS:
        f = os.path.join(a.results, m, "preds", f"ensemble_{m}_full.npz")
        if not os.path.exists(f):
            f = os.path.join(a.results, m, "preds", f"final_{m}_full_s0.npz")
        if os.path.exists(f):
            preds5[m] = align(np.load(f), te) * P["sigma"][te][:, None]     # next 5 daily log returns
            preds[m] = preds5[m][:, 0]
    preds_xs = {}
    for m in MODELS:
        f = os.path.join(a.results, m, "preds", f"ensemble_xs_{m}_full.npz")
        if not os.path.exists(f):
            f = os.path.join(a.results, m, "preds", f"xs_{m}_full_s0.npz")
        if os.path.exists(f):
            preds_xs[m] = align(np.load(f), te)[:, 0] * P["sigma"][te]
    # cross-architecture ensemble: average of the three market-neutral seed-ensembles (normalised outputs)
    combo_files = [os.path.join(a.results, m, "preds", f"ensemble_xs_{m}_full.npz") for m in MODELS]
    if all(os.path.exists(f) for f in combo_files):
        zz = [align(np.load(f), te) for f in combo_files]
        zz = [z / np.nanstd(z) for z in zz]           # put the three models on the same scale
        cp = np.mean(zz, 0) * np.mean([np.nanstd(align(np.load(f), te)) for f in combo_files])
        met, rp = evaluate(cp, te, p)
        out["summary"].append(rnd({"model": "combo", "features": "xs_full", "n_seeds": 9, "mean": {k: met.get(k) for k in KEY},
                                   "std": {k: 0.0 for k in KEY}, "ensemble": {k: met.get(k) for k in KEY}}))
        out["breakdown"]["combo:xs_full"] = rnd({"vol_tier": breakdown(rp[:, 0], te, p, "vol_tier"),
                                                 "asset_class": breakdown(rp[:, 0], te, p, "asset_class")})
        preds_xs["combo"] = cp[:, 0] * P["sigma"][te]
    rxf = os.path.join(a.results, "baselines", "preds", "ridge_xs_full.npz")
    if os.path.exists(rxf):
        preds_xs["ridge"] = align(np.load(rxf), te)[:, 0] * P["sigma"][te]
    rf = os.path.join(a.results, "baselines", "preds", "ridge_full.npz")
    if os.path.exists(rf):
        preds["ridge"] = align(np.load(rf), te)[:, 0] * P["sigma"][te]
    # ---------------- forecast cone: band width calibrated on the validation period only
    va = p.ends["val"]
    cq = cone_quantiles(P["y_ret"][va].astype(float), P["sigma"][va].astype(float))
    sig_te, y5 = P["sigma"][te].astype(float), P["y_ret"][te].astype(float)
    out["cone"] = {"levels": {str(L): {"lo": lst(lo), "hi": lst(hi)} for L, (lo, hi) in cq.items()},
                   "coverage": {m: {str(L): v for L, v in cone_coverage(preds5[m], y5, sig_te, cq).items()} for m in preds5},
                   "note": "band_k = predicted cumulative log return + q_k * sigma_t; q from 2022-23 validation"}
    nd = (dates + pd.offsets.BDay(1))
    for i, t in enumerate(tick):
        msk = tid == i
        if not msk.any():
            continue
        s = {"ticker": t, "dates": [d.strftime("%Y-%m-%d") for d in nd[msk]],
             "d0": [d.strftime("%Y-%m-%d") for d in dates[msk]],
             "actual": lst(c_next[msk]), "prev": lst(close_t[msk]),
             "pred": {m: lst(close_t[msk] * np.exp(v[msk])) for m, v in preds.items()},
             "pred_ret": {m: lst(v[msk] * 100) for m, v in preds.items()},
             "actual_ret": lst(y[msk] * 100),
             "sigma": lst(sig_te[msk], 6),
             "path": [lst(r, 2) for r in P["y_close"][te][msk]],
             "cum5": {m: [lst(r, 6) for r in np.cumsum(v[msk], 1)] for m, v in preds5.items()},
             "pred_xs": {m: lst(close_t[msk] * np.exp(v[msk])) for m, v in preds_xs.items()},
             "pred_ret_xs": {m: lst(v[msk] * 100) for m, v in preds_xs.items()}}
        json.dump(s, open(os.path.join(a.out, "series", f"{t}.json"), "w"))
    eq = {"dates": sorted({d.strftime("%Y-%m-%d") for d in dates})}
    dfe = pd.DataFrame({"d": dates, "y": y, "g": np.array(meta["region"])[tid]})
    eq["buy_hold"] = np.round(np.cumsum(dfe.groupby("d").y.mean().values) * 100, 3).tolist()
    for m, v in list(preds.items()) + [(k + "_xs", v) for k, v in preds_xs.items()]:
        v = np.nan_to_num(v)
        dfe["pos"] = np.sign(v)
        daily = (dfe.pos * dfe.y).groupby(dfe.d).mean()
        eq[f"{m}_dir"] = np.round(np.cumsum(daily.values) * 100, 3).tolist()
        dfe["p"] = v

        def ls(g):
            q = g.p.rank(pct=True)
            return g.y[q > 0.8].mean() - g.y[q <= 0.2].mean()
        lsr = dfe.groupby(["d", "g"]).apply(ls, include_groups=False).groupby(level=0).mean().fillna(0)
        eq[f"{m}_ls"] = np.round(np.cumsum(lsr.values) * 100, 3).tolist()
    out["equity"] = eq
    out["series_index"] = [{"ticker": t, "asset_class": meta["asset_class"][i], "vol_tier": meta["vol_tier"][i],
                            "region": meta["region"][i]} for i, t in enumerate(tick)]
    lf = os.path.join(a.results, "live_forecast.json")
    out["live"] = json.load(open(lf)) if os.path.exists(lf) else None
    json.dump(rnd(out), open(os.path.join(a.out, "summary.json"), "w"))
    print("exported", len(out["runs"]), "final runs,", len(out["search"]), "search runs ->", a.out)


if __name__ == "__main__":
    main()
