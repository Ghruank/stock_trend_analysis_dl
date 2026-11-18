"""Measure seconds/epoch for each architecture on the current device (used to size the run)."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, torch
from stockdl.data import load_panel, iterate
from stockdl.train import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
p = load_panel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset.npz"), "full", 96, dev)
print("train windows:", len(p.ends["train"]), "val:", len(p.ends["val"]), "test:", len(p.ends["test"]), "F:", p.X.shape[1])
cfgs = {"lstm": dict(hidden=128, layers=2), "transformer": dict(d_model=128, layers=3, n_heads=4, d_ff=256),
        "informer": dict(d_model=128, e_layers=3, n_heads=4, d_ff=256, factor=5)}
rng = np.random.default_rng(0)
for bs in (512, 1024):
    for name, cfg in cfgs.items():
        m = build_model(name, p, cfg)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
        it = iterate(p, "train", bs, True, rng)
        for i, b in enumerate(it):
            if i == 5:
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = torch.nn.functional.mse_loss(m(b).float(), b["y"])
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if i == 45:
                break
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = (time.time() - t0) / 40
        print(f"bs={bs} {name:12s} params={sum(x.numel() for x in m.parameters())/1e3:.0f}k  "
              f"{dt*1000:.0f} ms/step  -> {dt*math.ceil(len(p.ends['train'])/bs):.0f} s/epoch", flush=True)
