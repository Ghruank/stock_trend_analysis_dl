"""End-to-end experiment driver (resumable).

Phases
  baselines : naive random walk + ridge regression (non-deep reference points)
  search    : hyper-parameter search per architecture on the FULL feature set, selected by
              validation loss (test set is never looked at)
  final     : best config per architecture x feature sets {base, full} x N seeds,
              evaluated on the Jan 2024 - Nov 2026 test set, plus a seed-ensemble
  ablation  : best architecture, full features minus one feature group at a time

Usage:
  python run_experiments.py --phase all --device cuda --out results
  python run_experiments.py --phase search --smoke        # tiny CPU sanity run
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.data import load_panel  # noqa: E402
from stockdl.metrics import breakdown, evaluate, naive_metrics  # noqa: E402
from stockdl.train import fit, predict  # noqa: E402

LOOKBACK = 96
SEARCH_SPACE = {
    "lstm": {"hidden": [64, 128], "layers": [1, 2], "dropout": [0.1, 0.3], "lr": [1e-3, 3e-4]},
    "transformer": {"d_model": [64, 128], "layers": [2, 3], "dropout": [0.1, 0.3], "lr": [3e-4, 1e-4],
                    "n_heads": [4], "d_ff": [256]},
    "informer": {"d_model": [64, 128], "e_layers": [2, 3], "dropout": [0.1, 0.3], "lr": [3e-4, 1e-4],
                 "factor": [5], "n_heads": [4], "d_ff": [256], "attn": ["prob"]},
}
N_SEARCH = 6          # configs sampled per architecture
SEEDS = [0, 1, 2]
ABLATION_GROUPS = ["momentum", "candle", "volatility", "oscillator", "trend", "volume_liquidity",
                   "market", "cross_sectional", "calendar"]


def sample_configs(space, n, seed=0):
    keys = list(space)
    grid = [dict(zip(keys, v)) for v in itertools.product(*space.values())]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid), size=min(n, len(grid)), replace=False)
    return [grid[i] for i in sorted(idx)]


class Store:
    """Append-only JSONL store so a crashed / disconnected Colab session can resume."""

    def __init__(self, path):
        self.path = path
        self.rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []

    def has(self, key):
        return any(r["key"] == key for r in self.rows)

    def get(self, key):
        return next(r for r in self.rows if r["key"] == key)

    def add(self, row):
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


def run_baselines(args, store):
    if store.has("baseline:naive") and store.has("baseline:ridge"):
        return
    p = load_panel(args.data, "full", LOOKBACK, "cpu")
    te = p.ends["test"]
    store.add({"key": "baseline:naive", "phase": "baseline", "model": "naive", "features": "-",
               "test": naive_metrics(te, p)}) if not store.has("baseline:naive") else None
    if not store.has("baseline:ridge"):
        from sklearn.linear_model import Ridge
        X = p.X.numpy()
        Y = p.target.numpy()
        tr, va = p.ends["train"], p.ends["val"]
        best = None
        for a in [1, 10, 100, 1000, 10000]:
            m = Ridge(alpha=a).fit(X[tr], Y[tr])
            vl = float(np.mean((m.predict(X[va]) - Y[va]) ** 2))
            best = min(best or (vl, a, m), (vl, a, m), key=lambda z: z[0])
        vl, a, m = best
        pred = m.predict(X[te])
        met, rp = evaluate(pred, te, p)
        np.savez_compressed(os.path.join(args.out, "preds", "ridge_full.npz"), pred=pred.astype(np.float32), ends=te)
        store.add({"key": "baseline:ridge", "phase": "baseline", "model": "ridge", "features": "full",
                   "alpha": a, "val_loss": vl, "test": met,
                   "by_vol_tier": breakdown(rp[:, 0], te, p, "vol_tier"),
                   "by_asset_class": breakdown(rp[:, 0], te, p, "asset_class")})
    print("baselines done", flush=True)


def run_one(args, store, key, model, feats, cfg, seed, phase, panels, save_pred=False, target="ret"):
    if store.has(key):
        return store.get(key)
    pk = (feats, target)
    if pk not in panels:
        panels.clear()
        panels[pk] = load_panel(args.data, feats, LOOKBACK, args.device, target=target)
        panels[feats] = panels[pk]
        if args.smoke:
            for k in ("train", "val", "test"):
                panels[feats].ends[k] = panels[feats].ends[k][::40]
    p = panels[pk]
    print(f"=== {key}  cfg={cfg}", flush=True)
    m, info = fit(model, p, cfg, seed=seed, verbose=not args.quiet, max_minutes=args.max_minutes)
    pv = predict(m, p, "val")
    vmet, _ = evaluate(pv, p.ends["val"], p)
    row = {"key": key, "phase": phase, "model": model, "features": feats, "seed": seed, "cfg": cfg,
           "val": vmet, **{k: v for k, v in info.items()}}
    if phase != "search":
        pt = predict(m, p, "test")
        tmet, rp = evaluate(pt, p.ends["test"], p)
        row["test"] = tmet
        row["by_vol_tier"] = breakdown(rp[:, 0], p.ends["test"], p, "vol_tier")
        row["by_asset_class"] = breakdown(rp[:, 0], p.ends["test"], p, "asset_class")
        if phase in ("final", "xs") and seed == 0 and feats == "full":
            os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
            torch.save({"state_dict": m.state_dict(), "cfg": cfg, "model": model,
                        "n_features": int(p.X.shape[1]), "lookback": LOOKBACK},
                       os.path.join(args.out, "models", f"{model}_full_s0.pt" if phase == "final" else f"{model}_xs_full_s0.pt"))
        if save_pred:
            np.savez_compressed(os.path.join(args.out, "preds", key.replace(":", "_") + ".npz"),
                                pred=pt.astype(np.float32), ends=p.ends["test"])
    store.add(row)
    print(f"--> {key} val_loss={info['best_val_loss']:.4f} "
          + (f"test DA={row['test']['DA_%']:.2f} R2oos={row['test']['R2_oos_%']:+.3f} MAPE={row['test']['MAPE_%']:.3f}"
             if "test" in row else f"val R2oos={vmet['R2_oos_%']:+.3f} DA={vmet['DA_%']:.2f}"), flush=True)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def best_cfg(store, model):
    rows = [r for r in store.rows if r["phase"] == "search" and r["model"] == model]
    return min(rows, key=lambda r: r["best_val_loss"])["cfg"]


def ensemble(args, store, model, feats, prefix="final"):
    key = f"ensemble:{model}:{feats}" if prefix == "final" else f"ensemble_{prefix}:{model}:{feats}"
    if store.has(key):
        return
    files = [os.path.join(args.out, "preds", f"{prefix}_{model}_{feats}_s{s}.npz") for s in SEEDS]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 2:
        return
    p = load_panel(args.data, feats, LOOKBACK, "cpu")
    pred = np.mean([np.load(f)["pred"] for f in files], 0)
    met, rp = evaluate(pred, p.ends["test"], p)
    np.savez_compressed(os.path.join(args.out, "preds", key.split(":")[0] + f"_{model}_{feats}.npz"), pred=pred, ends=p.ends["test"])
    store.add({"key": key, "phase": "ensemble" if prefix == "final" else f"ensemble_{prefix}", "model": model, "features": feats, "n_seeds": len(files),
               "test": met, "by_vol_tier": breakdown(rp[:, 0], p.ends["test"], p, "vol_tier"),
               "by_asset_class": breakdown(rp[:, 0], p.ends["test"], p, "asset_class"),
               "by_ticker": breakdown(rp[:, 0], p.ends["test"], p, "ticker")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "baselines", "search", "final", "ablation", "xs"])
    ap.add_argument("--xs_features", default="full")
    ap.add_argument("--ablation_target", default="ret", choices=["ret", "xs"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "dataset.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="lstm,transformer,informer")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=25.0)
    ap.add_argument("--search_idx", default=None, help="comma list: run only these of the sampled configs (compute budget)")
    ap.add_argument("--patience", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    global N_SEARCH, SEEDS, ABLATION_GROUPS
    extra = {}
    if args.smoke:
        N_SEARCH, SEEDS, ABLATION_GROUPS = 1, [0], ["market"]
        extra = {"epochs": 1}
    if args.patience:
        extra["patience"] = args.patience
    store = Store(os.path.join(args.out, "runs.jsonl"))
    models = args.models.split(",")
    panels = {}
    t0 = time.time()

    if args.phase in ("all", "baselines"):
        run_baselines(args, store)

    if args.phase in ("all", "search"):
        for model in models:
            cfgs = list(enumerate(sample_configs(SEARCH_SPACE[model], N_SEARCH)))
            if args.search_idx:
                keep = {int(x) for x in args.search_idx.split(",")}
                cfgs = [c for c in cfgs if c[0] in keep]
            for i, cfg in cfgs:
                run_one(args, store, f"search:{model}:{i}", model, "full", {**cfg, **extra}, 0, "search", panels)

    if args.phase in ("all", "final"):
        for feats in ["full", "base"]:
            for model in models:
                cfg = {**best_cfg(store, model), **extra}
                for s in SEEDS:
                    run_one(args, store, f"final:{model}:{feats}:s{s}", model, feats, cfg, s, "final", panels,
                            save_pred=True)
        for feats in ["full", "base"]:
            for model in models:
                ensemble(args, store, model, feats)

    if args.phase == "xs":
        # Innovation: market-neutral target + early stopping on validation correlation
        for model in models:
            cfg = {**best_cfg(store, model), **extra, "monitor": "corr"}
            for feats in args.xs_features.split(","):
                for s in SEEDS:
                    run_one(args, store, f"xs:{model}:{feats}:s{s}", model, feats, cfg, s, "xs", panels,
                            save_pred=True, target="xs")
                ensemble(args, store, model, feats, prefix="xs")

    if args.phase in ("all", "ablation"):
        xs = args.ablation_target == "xs"
        ref_phase = "xs" if xs else "final"
        finals = [r for r in store.rows if r["phase"] == ref_phase and r["features"] == "full"]
        if finals:
            df = pd.DataFrame([{"model": r["model"], "v": r["best_val_loss"]} for r in finals])
            best_model = df.groupby("model").v.mean().idxmin()   # chosen on VALIDATION loss
            cfg = {**best_cfg(store, best_model), **extra, **({"monitor": "corr"} if xs else {})}
            for g in ABLATION_GROUPS:
                for s in SEEDS[:2]:
                    run_one(args, store, f"ablation{'_xs' if xs else ''}:{best_model}:drop:{g}:s{s}", best_model,
                            f"drop:{g}", cfg, s, "ablation_xs" if xs else "ablation", panels, target="xs" if xs else "ret")
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
