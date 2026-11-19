"""Render every figure used by the slides and the LaTeX report from the exported results.

python scripts/make_figures.py --data website/public/data --out report/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
C = {"lstm": "#2a78d6", "transformer": "#eb6834", "informer": "#1baf7a", "ridge": "#898781", "naive": "#c3c2b7", "combo": "#7a4fd0"}
N = {"lstm": "LSTM", "transformer": "Transformer", "informer": "Informer", "ridge": "Ridge", "naive": "Naive RW", "combo": "3-model ensemble"}
M = ["lstm", "transformer", "informer"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": .8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 11.5,
    "axes.titlelocation": "left", "legend.frameon": False, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def val(S, m, fs, k, ens=True):
    r = next((x for x in S["summary"] if x["model"] == m and x["features"] == fs), None)
    if r is None:
        return np.nan
    v = r["ensemble"][k] if ens and r.get("ensemble") else r["mean"][k]
    return np.nan if v is None else v


def sd(S, m, fs, k):
    r = next((x for x in S["summary"] if x["model"] == m and x["features"] == fs), None)
    return 0 if r is None or r["std"][k] is None else r["std"][k]


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("  ", name)


def bars(ax, labels, values, colors, fmt="{:.2f}", ref=None, err=None):
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=.62, yerr=err, error_kw={"ecolor": INK2, "capsize": 3, "lw": 1}, zorder=3)
    ax.set_xticks(x, labels)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=2)
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 3 if v >= (ref or 0) else -12), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)


# --------------------------------------------------------------------------------------------- figures
def fig_headline(S, out):
    ks = [("DA_%", "Directional accuracy (%)", 50, "{:.2f}"), ("IC", "Information coefficient", 0, "{:.4f}"),
          ("R2_oos_%", "Out-of-sample R² vs random walk (%)", 0, "{:.2f}"), ("Sharpe_LS", "Long/short Sharpe (gross)", 0, "{:.2f}")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    keys = M + ["ridge"]
    for ax, (k, t, ref, fmt) in zip(axs, ks):
        v = [val(S, m, "full", k) for m in M] + [S["baselines"].get("ridge", {}).get(k, np.nan)]
        e = [sd(S, m, "full", k) for m in M] + [0]
        bars(ax, [N[m] for m in keys], v, [C[m] for m in keys], fmt, ref)
        ax.set_title(t)
        ax.tick_params(axis="x", labelsize=9)
        if k == "DA_%":
            lo = np.nanmin(v)
            ax.set_ylim(min(48, lo - .5), max(53, np.nanmax(v) + .8))
    fig.suptitle("Test period Jan 2024 – Nov 2026 · full feature set · seed ensemble", x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_headline.png")


def fig_base_full(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, (k, t, ref) in zip(axs, [("DA_%", "Directional accuracy (%)", 50), ("IC", "Information coefficient", 0), ("Sharpe_LS", "Long/short Sharpe", 0)]):
        x = np.arange(3)
        b = [val(S, m, "base", k, False) for m in M]
        f = [val(S, m, "full", k, False) for m in M]
        ax.bar(x - .19, b, .36, color=[C[m] for m in M], alpha=.38, label="Base (8 features)", zorder=3)
        ax.bar(x + .19, f, .36, color=[C[m] for m in M], label="Full (87 features)", zorder=3)
        ax.set_xticks(x, [N[m] for m in M])
        ax.axhline(ref, color=MUTED, lw=1, ls="--")
        if k == "DA_%":
            ax.set_ylim(min(48, np.nanmin(b + f) - .5), max(53, np.nanmax(b + f) + .5))
        ax.set_title(t)
    axs[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_base_vs_full.png")


def fig_innovation(S, out):
    """Target engineering: raw volatility-normalised return target vs market-neutral (cross-sectional) target."""
    if not any(x["features"] == "xs_full" for x in S["summary"]):
        return
    ks = [("IC", "Information coefficient (rank corr.)", "{:.4f}"), ("DA_xs_%", "Beat-the-peers accuracy (%)", "{:.2f}"),
          ("Sharpe_LS", "Long/short Sharpe (gross)", "{:.2f}")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.7))
    models = M + (["combo"] if any(x["model"] == "combo" for x in S["summary"]) else [])
    for ax, (k, t, fmt) in zip(axs, ks):
        x = np.arange(len(models))
        a = [val(S, m, "full", k) if m != "combo" else np.nan for m in models]
        b = [val(S, m, "xs_full", k) for m in models]
        ax.bar(x - .19, a, .36, color=[C[m] for m in models], alpha=.35, zorder=3, label="Raw return target")
        ax.bar(x + .19, b, .36, color=[C[m] for m in models], zorder=3, label="Market-neutral target")
        for i, v in enumerate(b):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (i + .19, v), xytext=(0, 3 if v >= 0 else -12), textcoords="offset points",
                            ha="center", fontsize=8.5)
        rx = S["baselines"].get("ridge_xs", {}).get(k)
        ref = 50 if k == "DA_xs_%" else 0
        ax.axhline(ref, color=MUTED, lw=1)
        if rx is not None:
            ax.axhline(rx, color=C["ridge"], lw=1.2, ls="--", label="Ridge (market-neutral)")
        ax.set_xticks(x, [N[m].replace("3-model ensemble", "Ensemble") for m in models])
        if k == "DA_xs_%":
            vv = [v for v in a + b + [rx or 50] if np.isfinite(v)]
            ax.set_ylim(min(49, min(vv) - .3), max(vv) + .5)
        ax.set_title(t)
    axs[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Innovation: predict which asset beats its peers, not where the market goes · test 2024–2026",
                 x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_innovation.png")


def fig_ridge_ablation(S, out):
    R = S.get("ridge_ablation")
    if not R:
        return
    g = pd.DataFrame(R["groups"])
    g["dic"] = g.IC - R["ref"]["IC"]
    g["lab"] = ["only the 8 base features" if r.n > 60 else f"drop {r.group.replace('_', ' ')} ({r.n})" for r in g.itertuples()]
    g = g.sort_values("dic")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(g.lab, g.dic * 1000, color=np.where(g.dic < 0, C["lstm"], "#c3c2b7"), zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ test IC ×1000 vs all 87 features  (blue = removing it hurts → family carries signal)")
    ax.set_title(f"Which feature families matter? (linear probe, full IC = {R['ref']['IC']:.4f})")
    save(fig, out, "fig_ridge_ablation.png")


def fig_ablation(S, out):
    A = S.get("ablation")
    if not A:
        return
    g = pd.DataFrame([{"group": x["group"].replace("_", " "), "dv": x["delta_val_loss"] * 1000,
                       "dic": x["test"]["IC"] - A["ref_test"]["IC"], "n": x["n_features_dropped"]} for x in A["groups"]]).sort_values("dv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": .45})
    for ax, k, t in [(axs[0], "dv", "Δ validation loss ×1000 when removed\n(> 0 = family helps)"),
                     (axs[1], "dic", "Δ test IC when removed\n(< 0 = family helps)")]:
        good = g[k] > 0 if k == "dv" else g[k] < 0
        ax.barh(g.group, g[k], color=np.where(good, C["lstm"], "#c3c2b7"), zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_title(t, fontsize=10.5)
    fig.suptitle(f"Feature-family ablation ({N[A['model']]}, 2 seeds each)", x=0.01, ha="left", fontweight="bold", y=1.04)
    save(fig, out, "fig_ablation.png")


def fig_breakdown(S, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, by, order, lab in [(axs[0], "vol_tier", ["low", "mid", "high"], ["Low vol", "Mid vol", "High vol"]),
                               (axs[1], "asset_class", ["equity_in", "equity_us", "commodity"], ["Indian equity", "US equity", "Commodity"])]:
        x = np.arange(len(order))
        w = .2
        series = M + ["ridge"]
        for i, m in enumerate(series):
            src = S["breakdown"].get(f"{m}:full") if m != "ridge" else S.get("baseline_breakdown", {}).get("ridge")
            if not src:
                continue
            v = [src[by].get(g, {}).get("DA_%", np.nan) for g in order]
            ax.bar(x + (i - 1.5) * w, v, w * .92, color=C[m], label=N[m], zorder=3)
        ax.axhline(50, color=MUTED, ls="--", lw=1)
        ax.set_xticks(x, lab)
        ax.set_ylim(46, 56)
        ax.set_title("Directional accuracy (%) by " + ("volatility tier" if by == "vol_tier" else "asset class"))
    axs[0].legend(ncol=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_breakdown.png")


def cone_at(s, Q, m, i, levels=("80", "95")):
    """Fan for anchor row i: dates, centre path and bands (prices), anchored on the close at t."""
    c0, sg, cum = s["prev"][i], s["sigma"][i], np.r_[0, s["cum5"][m][i]]
    d = pd.to_datetime([s["d0"][i]] + s["d0"][i + 1:i + 6])
    band = {L: (c0 * np.exp(cum + np.r_[0, Q[L]["lo"]] * sg), c0 * np.exp(cum + np.r_[0, Q[L]["hi"]] * sg)) for L in levels}
    return d, c0 * np.exp(cum), band


def fig_cone(S, data_dir, out, m="lstm"):
    import matplotlib.dates as mdates
    Q = S["cone"]["levels"]
    cov = S["cone"]["coverage"][m]
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.55, wspace=.18)
    # --- top: a chain of non-overlapping 5-day cones over the last 6 months of the test set
    ax = fig.add_subplot(gs[0, :])
    s = json.load(open(os.path.join(data_dir, "series", "TCS.json")))
    n = len(s["d0"])
    lo_i = n - 6 - 125
    d_all = pd.to_datetime(s["d0"])
    for k, i in enumerate(range(lo_i, n - 6, 5)):
        d, cen, band = cone_at(s, Q, m, i)
        ax.fill_between(d, *band["95"], color=C[m], alpha=.13, lw=0, label="95% band" if k == 0 else None)
        ax.fill_between(d, *band["80"], color=C[m], alpha=.28, lw=0, label="80% band" if k == 0 else None)
        ax.plot(d, cen, color=C[m], lw=1.3, ls="--", label=f"{N[m]} forecast path" if k == 0 else None)
        ax.plot(d[0], cen[0], "o", color=C[m], ms=3.5)
    ax.plot(d_all[lo_i:n], s["prev"][lo_i:n], color=INK, lw=1.7, label="Actual close", zorder=5)
    ax.set_title(f"TCS: a new 5-day forecast cone every week, actual price drawn over it (last 6 months of the test set)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    ax.text(1, -.16, f"Actual price inside the 80% band {cov['80']['all']:.1f}% of the time, inside the 95% band "
            f"{cov['95']['all']:.1f}% (all 49 assets, Jan 2024 – Nov 2026)", transform=ax.transAxes, ha="right",
            fontsize=9, color=INK2)
    # --- bottom: three single cones, zoomed, for a low-, mid- and high-volatility asset
    for col, t in enumerate(["KO", "GOLD", "YESBANK"]):
        ax = fig.add_subplot(gs[1, col])
        s = json.load(open(os.path.join(data_dir, "series", f"{t}.json")))
        n = len(s["d0"])
        i = n - 6 - 15
        d, cen, band = cone_at(s, Q, m, i)
        h0 = max(0, i - 30)
        ax.plot(pd.to_datetime(s["d0"][h0:i + 1]), s["prev"][h0:i + 1], color=INK, lw=1.6)
        ax.fill_between(d, *band["95"], color=C[m], alpha=.13, lw=0)
        ax.fill_between(d, *band["80"], color=C[m], alpha=.28, lw=0)
        ax.plot(d, cen, color=C[m], lw=1.4, ls="--")
        ax.plot(d, [s["prev"][i]] + s["path"][i], color="#d03b3b", lw=1.8, marker="o", ms=3.5, label="what actually happened")
        ax.axvline(d[0], color=MUTED, lw=.8, ls=":")
        tier = S["meta"]["vol_tier"][S["meta"]["tickers"].index(t)]
        ax.set_title(f"{t} · {tier} volatility\nforecast made {d[0]:%d %b %Y}", fontsize=10)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=8.5)
        if col == 0:
            ax.legend(fontsize=8.5, loc="upper left")
    save(fig, out, "fig_cone.png")


def fig_cone_deck(S, data_dir, out, m="lstm", t="TCS"):
    """Slide-sized version: one asset, weekly cones over the last ~4 months, large type."""
    import matplotlib.dates as mdates
    Q = S["cone"]["levels"]
    s = json.load(open(os.path.join(data_dir, "series", f"{t}.json")))
    n = len(s["d0"])
    lo_i = n - 6 - 80
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for k, i in enumerate(range(lo_i, n - 6, 5)):
        d, cen, band = cone_at(s, Q, m, i)
        ax.fill_between(d, *band["95"], color=C[m], alpha=.14, lw=0, label="95% range" if k == 0 else None)
        ax.fill_between(d, *band["80"], color=C[m], alpha=.32, lw=0, label="80% range" if k == 0 else None)
        ax.plot(d, cen, color=C[m], lw=1.6, ls="--", label=f"{N[m]} forecast" if k == 0 else None)
    ax.plot(pd.to_datetime(s["d0"][lo_i:n]), s["prev"][lo_i:n], color=INK, lw=2.2, label="actual price", zorder=5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(labelsize=12)
    ax.legend(ncol=2, fontsize=11.5, loc="upper left")
    ax.set_title(f"{t}: a 5-day forecast cone every week, real price on top", fontsize=14)
    save(fig, out, "fig_cone_deck.png")


def fig_equity(S, out):
    E = S["equity"]
    d = pd.to_datetime(E["dates"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    import matplotlib.dates as mdates
    for ax, kind, t in [(axs[0], "ls", "Long/short: top vs bottom 20%"), (axs[1], "dir", "Long or short each asset by forecast sign")]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        xs = any(k.endswith(f"_xs_{kind}") for k in E)
        for m in M + ["combo", "ridge"]:
            key = f"{m}_xs_{kind}" if xs and f"{m}_xs_{kind}" in E else f"{m}_{kind}"
            if key in E:
                ax.plot(d, E[key], color=C[m], lw=1.8 if m != "ridge" else 1.3, ls="-" if m != "ridge" else "--", label=N[m])
        if kind == "dir":
            ax.plot(d, E["buy_hold"], color=INK, lw=1.2, ls=":", label="Equal-weight buy & hold")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(t + "\ncumulative %, before costs", fontsize=10.5)
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig_equity.png")


def fig_examples(S, data_dir, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, t in zip(axs, ["TCS", "GOLD", "TSLA"]):
        s = json.load(open(os.path.join(data_dir, "series", f"{t}.json")))
        d = pd.to_datetime(s["dates"])[-90:]
        ax.plot(d, s["actual"][-90:], color=INK, lw=1.8, label="Actual")
        for m in M:
            if s["pred"].get(m):
                ax.plot(d, s["pred"][m][-90:], color=C[m], lw=1.2, label=N[m])
        ax.set_title(f"{t} — next-day close, last 90 test days")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8.5)
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig_examples.png")


def fig_training(S, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
    for ax, m in zip(axs, M):
        rs = [r for r in S["search"] if r["model"] == m]
        if not rs:
            continue
        best = min(rs, key=lambda r: r["val_loss"])
        for r in rs:
            h = pd.DataFrame(r["history"])
            ax.plot(h.epoch + 1, h.val_loss, color=C[m], alpha=1 if r is best else .3, lw=2.2 if r is best else 1.2)
        ax.set_title(f"{N[m]} — validation loss per epoch ({len(rs)} configs)")
        ax.set_xlabel("epoch")
    fig.tight_layout()
    save(fig, out, "fig_training.png")


def fig_live(S, out):
    L = S.get("live")
    if not L:
        return
    pick = [a for a in L["assets"] if a["ticker"] in ("TCS", "RELIANCE", "GOLD", "NVDA")][:4]
    fig, axs = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.3))
    for ax, a in zip(np.atleast_1d(axs), pick):
        h = pd.DataFrame(a["history"], columns=["d", "c"])
        h["d"] = pd.to_datetime(h["d"])
        ax.plot(h.d[-40:], h.c[-40:], color=INK, lw=1.8)
        fd = pd.to_datetime([a["last_date"]] + a["future_dates"])
        for m in M:
            if m in a["forecasts"]:
                ax.plot(fd, [a["last_close"]] + a["forecasts"][m]["path"], color=C[m], lw=1.8, ls="--", marker="o", ms=3,
                        label=f"{N[m]} {a['forecasts'][m]['ret_5d_%']:+.1f}%")
        ax.axvline(fd[0], color=MUTED, lw=1, ls=":")
        ax.set_title(f"{a['ticker']} · next 5 days")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig_live.png")


def box(ax, x, y, w, h, title, sub, fc="#f2f1ed", ec="#c3c2b7", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * .66, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * .3, sub, ha="center", va="center", fontsize=8.3, color=INK2, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.4})


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    steps = [("Raw snapshot", "yfinance · 49 assets\n+7 market series\n2005 → Nov 2026"),
             ("Cleaning", "bad ticks, demergers,\nfutures rolls"),
             ("87 features", "10 families, no\nlook-ahead"),
             ("Scaling", "per asset, fitted\non 2007–2021"),
             ("Windows", "96 days → next 5\nreturns / σ"),
             ("3 models", "LSTM · Transformer\n· Informer"),
             ("Evaluation", "2024–26 test: DA, IC,\nR²oos, MAPE, Sharpe")]
    w, gap = 1.8, .33
    for i, (t, s) in enumerate(steps):
        x = .1 + i * (w + gap)
        hl = i in (2, 5)
        box(ax, x, .9, w, 1.5, t, s, fc="#e8f1fc" if hl else "#f2f1ed", ec=C["lstm"] if hl else "#c3c2b7")
        if i < len(steps) - 1:
            arrow(ax, x + w + .02, 1.65, x + w + gap - .02, 1.65)
    ax.text(.1, 2.85, "Leakage-free pipeline: everything left of the models is computed once, locally, and frozen",
            fontsize=10, color=INK2)
    ax.text(.1 + 4 * (w + gap), .45, "chronological: train 2007–21 · val 2022–23 · test 2024–26 · purge gap 5 days",
            fontsize=8.8, color=INK2, ha="left")
    save(fig, out, "fig_pipeline.png")


def fig_architectures(out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = {
        "LSTM": [("Input 96×87", ""), ("Linear + asset emb.", ""), ("2-layer LSTM", "gated memory"), ("Last state ‖ attention pool", ""), ("MLP head → 5 returns", "")],
        "Transformer": [("Input 96×87", ""), ("Linear + Time2Vec + asset", "learnable time encoding"), ("Pre-LN encoder × L", "multi-head self-attention"), ("Last ‖ mean pool", ""), ("MLP head → 5 returns", "")],
        "Informer": [("Encoder 96×87", "conv token + temporal emb."), ("ProbSparse attention", "top-u queries, O(L log L)"), ("Distilling conv ÷2", "halves the sequence"), ("Decoder: 48-day start token", "+ cross-attention"), ("Projection → 5 returns", "one shot, no recursion")],
    }
    cols = {"LSTM": C["lstm"], "Transformer": C["transformer"], "Informer": C["informer"]}
    for ax, (name, layers) in zip(axs, specs.items()):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        ax.text(2, 5.95, name, ha="center", fontsize=13, fontweight="bold", color=cols[name])
        for i, (t, s) in enumerate(layers):
            y = 4.8 - i * 1.15
            ax.add_patch(FancyBboxPatch((.3, y), 3.4, .82, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="white", ec=cols[name], lw=1.5))
            ax.text(2, y + (.53 if s else .41), t, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
            if s:
                ax.text(2, y + .22, s, ha="center", va="center", fontsize=8.3, color=INK2)
            if i < len(layers) - 1:
                arrow(ax, 2, y - .02, 2, y - .31)
    fig.tight_layout()
    save(fig, out, "fig_architectures.png")


def fig_features(S, out):
    from collections import Counter
    cnt = Counter(f["group"] for f in S["meta"]["features"])
    order = sorted(cnt, key=lambda g: -cnt[g])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([g.replace("_", " ") for g in order][::-1], [cnt[g] for g in order][::-1],
            color=["#c3c2b7" if g == "base" else C["lstm"] for g in order][::-1], zorder=3)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    for i, g in enumerate(order[::-1]):
        ax.text(cnt[g] + .2, i, str(cnt[g]), va="center", fontsize=9)
    ax.set_title("87 engineered features in 10 families (grey = 8 basic price/volume features)")
    save(fig, out, "fig_features.png")


def fig_deciles(S, data_dir, out):
    rows = []
    for t in S["meta"]["tickers"]:
        s = json.load(open(os.path.join(data_dir, "series", f"{t}.json")))
        for m in M:
            src = s.get("pred_ret_xs", {}) if s.get("pred_ret_xs", {}).get(m) else s["pred_ret"]
            if src.get(m):
                rows.append(pd.DataFrame({"m": m, "d": s["dates"], "p": src[m], "y": s["actual_ret"]}))
    if not rows:
        return
    df = pd.concat(rows).dropna()
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.3))
    fig.suptitle("Market-neutral models, test 2024–26: a rising staircase means the daily ranking works", x=0.01, ha="left", color=INK2, fontsize=10, y=1.04)
    for ax, m in zip(axs, M):
        g = df[df.m == m].copy()
        g["q"] = g.groupby("d").p.rank(pct=True)
        g["dec"] = np.minimum((g.q * 5).astype(int), 4) + 1
        r = g.groupby("dec").y.mean() * 100
        ax.bar(r.index, r.values, color=[C[m] if v > 0 else "#c3c2b7" for v in r.values], zorder=3)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(1, 6), ["Q1\nbearish", "Q2", "Q3", "Q4", "Q5\nbullish"])
        ax.set_title(f"{N[m]} · next-day return by forecast quintile (bps)", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig_quintiles.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "website", "public", "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(os.path.join(a.data, "summary.json")))
    fig_pipeline(a.out)
    fig_architectures(a.out)
    fig_features(S, a.out)
    fig_headline(S, a.out)
    fig_base_full(S, a.out)
    fig_ablation(S, a.out)
    fig_innovation(S, a.out)
    fig_ridge_ablation(S, a.out)
    fig_breakdown(S, a.out)
    fig_cone(S, a.data, a.out)
    fig_cone_deck(S, a.data, a.out)
    fig_equity(S, a.out)
    fig_examples(S, a.data, a.out)
    fig_training(S, a.out)
    fig_live(S, a.out)
    fig_deciles(S, a.data, a.out)


if __name__ == "__main__":
    main()
