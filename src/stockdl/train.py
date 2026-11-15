"""Model factory + training loop (early stopping on validation loss, warmup + cosine LR)."""
from __future__ import annotations

import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("LSTM", "Transformer", "Informer"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from lstm_model import LSTMForecaster  # noqa: E402
from transformer_model import TransformerForecaster  # noqa: E402
from informer_model import InformerForecaster  # noqa: E402

from .data import iterate  # noqa: E402

MODELS = {"lstm": LSTMForecaster, "transformer": TransformerForecaster, "informer": InformerForecaster}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, panel, cfg: dict) -> nn.Module:
    kw = {k: v for k, v in cfg.items() if k not in ("lr", "weight_decay", "batch_size", "epochs", "patience", "monitor")}
    return MODELS[name](n_features=panel.X.shape[1], n_assets=len(panel.meta["tickers"]),
                        horizon=panel.horizon, lookback=panel.lookback, **kw).to(panel.X.device)


@torch.no_grad()
def predict(model, panel, split, batch_size=2048, label_len=48):
    model.eval()
    out = []
    amp = panel.X.is_cuda
    for b in iterate(panel, split, batch_size, shuffle=False, label_len=label_len):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(model(b).float().cpu())
    return torch.cat(out).numpy()


def fit(name: str, panel, cfg: dict, seed: int = 0, verbose: bool = True, max_minutes: float = 30.0):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name, panel, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr, bs = cfg.get("lr", 1e-3), cfg.get("batch_size", 512)
    epochs, patience = cfg.get("epochs", 20), cfg.get("patience", 4)
    label_len = cfg.get("label_len", 48)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
    steps_per_epoch = math.ceil(len(panel.ends["train"]) / bs)
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    loss_fn = nn.MSELoss()
    amp = panel.X.is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for b in iterate(panel, "train", bs, shuffle=True, rng=rng, label_len=label_len):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(b)
            loss = loss_fn(out.float(), b["y"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(b["y"])
            n += len(b["y"])
        pv = predict(model, panel, "val", label_len=label_len)
        yv = panel.target[torch.as_tensor(panel.ends["val"], device=panel.X.device)].cpu().numpy()
        vl = float(np.mean((pv - yv) ** 2))
        v_r2 = float(1 - np.sum((pv[:, 0] - yv[:, 0]) ** 2) / np.sum(yv[:, 0] ** 2))
        v_corr = float(np.corrcoef(pv[:, 0], yv[:, 0])[0, 1]) if np.std(pv[:, 0]) > 0 else 0.0
        hist.append({"epoch": ep, "train_loss": tl / n, "val_loss": vl, "val_r2_1d": v_r2, "val_corr": v_corr})
        if cfg.get("monitor") == "corr":
            vl = -v_corr if np.isfinite(v_corr) else float("inf")   # lower is better
        if verbose:
            print(f"  [{name}] ep {ep:02d} train {tl / n:.4f} val {vl:.4f} valR2 {v_r2 * 100:+.2f}% corr {v_corr:+.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if not np.isfinite(vl):
            bad = patience
        if np.isfinite(vl) and vl < best - 1e-5:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
        if (time.time() - t0) / 60 > max_minutes:
            print("  time budget reached", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "epochs_run": len(hist), "n_params": n_params,
                   "train_minutes": (time.time() - t0) / 60, "history": hist}
