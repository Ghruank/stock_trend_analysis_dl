"""Add metrics introduced after the first runs finished (DA_xs_%) to results/*/runs.jsonl, from the saved predictions."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stockdl.data import load_panel  # noqa: E402
from stockdl.metrics import evaluate  # noqa: E402

p = load_panel(os.path.join(ROOT, "data", "dataset.npz"), "full", 96, "cpu")
te = p.ends["test"]


def preds(path):
    z = np.load(path)
    out = np.full((len(te), z["pred"].shape[1]), np.nan, np.float32)
    pos = {e: i for i, e in enumerate(te)}
    for e, row in zip(z["ends"], z["pred"]):
        out[pos[e]] = row
    return out


for m in ["lstm", "transformer", "informer", "baselines"]:
    f = os.path.join(ROOT, "results", m, "runs.jsonl")
    rows = [json.loads(l) for l in open(f)]
    n = 0
    for r in rows:
        t = r.get("test")
        if not t or t.get("DA_xs_%") is not None:
            continue
        ph, fs = r.get("phase"), r.get("features")
        if ph == "final":
            name = f"final_{r['model']}_{fs}_s{r['seed']}.npz"
        elif ph == "ensemble":
            name = f"ensemble_{r['model']}_{fs}.npz"
        elif r.get("key") == "baseline:ridge":
            name = "ridge_full.npz"
        else:
            continue
        path = os.path.join(ROOT, "results", m, "preds", name)
        if os.path.exists(path):
            t["DA_xs_%"] = evaluate(preds(path), te, p)[0]["DA_xs_%"]
            n += 1
    with open(f, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(m, "backfilled", n)
