"""Generate LaTeX tables + number macros for the report from website/public/data/summary.json.

python scripts/make_report_tables.py  ->  report/generated/*.tex
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S = json.load(open(os.path.join(ROOT, "website", "public", "data", "summary.json")))
OUT = os.path.join(ROOT, "report", "generated")
os.makedirs(OUT, exist_ok=True)
M = ["lstm", "transformer", "informer"]
N = {"lstm": "LSTM", "transformer": "Transformer", "informer": "Informer", "combo": "3-model ensemble",
     "ridge": "Ridge (linear)", "ridge_xs": "Ridge (linear)", "naive": "Naive random walk"}


def summ(m, fs):
    return next((x for x in S["summary"] if x["model"] == m and x["features"] == fs), None)


def get(m, fs, k, ens=True):
    r = summ(m, fs)
    if r is None:
        return None
    src = r["ensemble"] if ens and r.get("ensemble") else r["mean"]
    return src.get(k)


def f(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{d}f}"


COLS = [("DA_%", "DA\\,\\%", 2), ("DA_xs_%", "DA$_{xs}$\\,\\%", 2), ("IC", "IC", 4), ("R2_oos_%", "$R^2_{oos}$\\,\\%", 3),
        ("MAPE_%", "MAPE\\,\\%", 3), ("Sharpe_LS", "SR$_{LS}$", 2), ("Sharpe_dir_net", "SR$_{dir,net}$", 2), ("DA_5d_%", "DA$_{5d}$\\,\\%", 2)]


def table(rows, caption, label, note=""):
    best = {}
    for k, _, _ in COLS:
        vals = [r[1](k) for r in rows if r[1](k) is not None and r[0] not in ("naive",)]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            best[k] = min(vals) if k == "MAPE_%" else max(vals)
    s = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "r" * len(COLS) + "}", "\\toprule",
         "Model & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
    for key, fn, name in rows:
        cells = []
        for k, _, d in COLS:
            v = fn(k)
            t = f(v, d)
            if v is not None and k in best and np.isfinite(v) and abs(v - best[k]) < 1e-12 and key != "naive":
                t = "\\textbf{" + t + "}"
            cells.append(t)
        s.append(name + " & " + " & ".join(cells) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}}"]
    if note:
        s.append("\\\\[2pt]\\parbox{0.97\\linewidth}{\\footnotesize " + note + "}")
    s.append("\\end{table}")
    return "\n".join(s)


def write(name, txt):
    open(os.path.join(OUT, name), "w").write(txt + "\n")
    print("  ", name)


B = S["baselines"]
note = ("DA = directional accuracy of the next-day return sign; DA$_{xs}$ = accuracy of beating vs.\\ lagging the regional "
        "peer average; IC = mean daily rank correlation; $R^2_{oos}$ vs.\\ zero-return forecast; MAPE on next-day close; "
        "SR$_{LS}$ = annualised Sharpe of the daily top-20\\%/bottom-20\\% long/short book (gross); SR$_{dir,net}$ = per-asset "
        "sign strategy after 10\\,bp costs. Bold = best non-naive value.")

# --- main table (raw return target, full features)
rows = [(m, (lambda k, m=m: get(m, "full", k)), N[m]) for m in M]
rows += [("ridge", lambda k: B.get("ridge", {}).get(k), N["ridge"]), ("naive", lambda k: B.get("naive", {}).get(k), N["naive"])]
write("tab_main.tex", table(rows, "Test set (Jan 2024 -- Nov 2026), raw volatility-normalised return target, 87 features, "
                                  "3-seed ensemble.", "tab:main", note))

# --- market-neutral table
if any(x["features"] == "xs_full" for x in S["summary"]):
    rows = [(m, (lambda k, m=m: get(m, "xs_full", k)), N[m]) for m in M if summ(m, "xs_full")]
    if summ("combo", "xs_full"):
        rows.append(("combo", lambda k: get("combo", "xs_full", k), "\\textit{3-model ensemble}"))
    rows += [("ridge", lambda k: B.get("ridge_xs", {}).get(k), N["ridge"]), ("naive", lambda k: B.get("naive", {}).get(k), N["naive"])]
    write("tab_xs.tex", table(rows, "Test set, \\textbf{market-neutral target} (returns demeaned within region and day), "
                                    "checkpoint selected on validation correlation, 3-seed ensemble.", "tab:xs", note))
else:
    write("tab_xs.tex", "% market-neutral runs not available")

# --- feature-set table (seed mean +- sd)
lines = ["\\begin{table}[H]\\centering\\small",
         "\\caption{Effect of the input representation (mean $\\pm$ s.d.\\ over 3 seeds).}\\label{tab:fs}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrr}", "\\toprule", "Model & Inputs & DA\\,\\% & IC & MAPE\\,\\% & SR$_{LS}$ \\\\", "\\midrule"]
FSN = {"base": "Base 8 (stationary)", "full": "Full 87", "xs_full": "Full 87 + market-neutral",
       "xs_base": "Base 8 + market-neutral"}
for m in M:
    for fs in ["base", "full", "xs_base", "xs_full"]:
        r = summ(m, fs)
        if not r:
            continue
        cells = []
        for k, d in [("DA_%", 2), ("IC", 4), ("MAPE_%", 3), ("Sharpe_LS", 2)]:
            mu, sd = r["mean"].get(k), r["std"].get(k)
            cells.append(f(mu, d) + ("" if not sd else f" $\\pm$ {f(sd, d)}"))
        lines.append(f"{N[m]} & {FSN[fs]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
lines[-1] = "\\bottomrule"
lines += ["\\end{tabular}}", "\\end{table}"]
write("tab_fs.tex", "\n".join(lines))

# --- hyper-parameter search table
lines = ["\\begin{table}[H]\\centering\\small", "\\caption{Hyper-parameter search (validation 2022--2023). Selected configuration in bold.}\\label{tab:search}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrr}", "\\toprule", "Model & Configuration & Params & Epochs & Val.\\ loss \\\\", "\\midrule"]
for m in M:
    rs = sorted([r for r in S["search"] if r["model"] == m], key=lambda r: r["val_loss"])
    for i, r in enumerate(rs):
        cfg = ", ".join(f"{k}={v}" for k, v in r["cfg"].items() if k not in ("epochs", "patience", "batch_size"))
        cfg = cfg.replace("_", "\\_")
        row = f"{N[m]} & {{\\scriptsize {cfg}}} & {r.get('params', 0) / 1e3:.0f}k & {r.get('epochs', '')} & {r['val_loss']:.4f} \\\\"
        lines.append(row.replace(f"{r['val_loss']:.4f}", f"\\textbf{{{r['val_loss']:.4f}}}") if i == 0 else row)
    lines.append("\\midrule")
lines[-1] = "\\bottomrule"
lines += ["\\end{tabular}}", "\\end{table}"]
write("tab_search.tex", "\n".join(lines))

# --- macros
mac = {}
for m in M:
    for fs, tag in [("full", "Raw"), ("xs_full", "Xs"), ("base", "Base")]:
        for k, kk, d in [("DA_%", "DA", 2), ("IC", "IC", 4), ("Sharpe_LS", "LS", 2), ("MAPE_%", "MAPE", 2), ("DA_xs_%", "DAxs", 2),
                         ("R2_oos_%", "Rtwo", 3)]:
            mac[f"{N[m]}{tag}{kk}"] = f(get(m, fs, k), d)
for k, kk, d in [("IC", "IC", 4), ("Sharpe_LS", "LS", 2), ("DA_xs_%", "DAxs", 2), ("DA_%", "DA", 2)]:
    mac[f"Combo{kk}"] = f(get("combo", "xs_full", k), d)
    mac[f"Ridge{kk}"] = f(B.get("ridge", {}).get(k), d)
    mac[f"RidgeXs{kk}"] = f(B.get("ridge_xs", {}).get(k), d)
mac["NaiveMAPE"] = f(B["naive"]["MAPE_%"], 2)
mac["NAssets"] = str(len(S["meta"]["tickers"]))
mac["NFeat"] = str(S["meta"]["n_features"]["full"])
mac["NTrain"] = f"{S['meta']['rows_valid_per_split']['train']:,}".replace(",", "{,}")
mac["NVal"] = f"{S['meta']['rows_valid_per_split']['val']:,}".replace(",", "{,}")
mac["NTest"] = f"{S['meta']['rows_valid_per_split']['test']:,}".replace(",", "{,}")
cv = S.get("cone", {}).get("coverage", {}).get("lstm")
if cv:
    mac["ConeEighty"] = f(cv["80"]["all"], 1)
    mac["ConeNinetyFive"] = f(cv["95"]["all"], 1)
    mac["ConeHalfWidth"] = f(cv["80"]["width_5d_%"] / 2, 1)
mac["LiveDate"] = (S.get("live") or {}).get("generated_for", "--")
write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()))


# ------------------------------------------------------------------ markdown tables -> READMEs
MDCOLS = [("DA_%", "DA %", 2), ("DA_xs_%", "Peer-DA %", 2), ("IC", "IC", 4), ("R2_oos_%", "R²oos %", 3), ("MAPE_%", "MAPE %", 3),
          ("Sharpe_LS", "L/S Sharpe", 2), ("Sharpe_dir_net", "Dir. Sharpe net", 2)]


def md(rows):
    out = ["| Model | " + " | ".join(c[1] for c in MDCOLS) + " |", "|---|" + "---:|" * len(MDCOLS)]
    for name, fn in rows:
        out.append(f"| {name} | " + " | ".join(f(fn(k), d) for k, _, d in MDCOLS) + " |")
    return "\n".join(out)


def md_main():
    t = ["**Raw volatility-normalised return target, 87 features, 3-seed ensemble (test Jan 2024 – Nov 2026)**", "",
         md([(N[m], (lambda k, m=m: get(m, "full", k))) for m in M] +
            [("Ridge (linear)", lambda k: B.get("ridge", {}).get(k)), ("Naive random walk", lambda k: B["naive"].get(k))])]
    if any(x["features"] == "xs_full" for x in S["summary"]):
        rows = [(N[m], (lambda k, m=m: get(m, "xs_full", k))) for m in M if summ(m, "xs_full")]
        if summ("combo", "xs_full"):
            rows.append(("**3-model ensemble**", lambda k: get("combo", "xs_full", k)))
        rows.append(("Ridge (linear)", lambda k: B.get("ridge_xs", {}).get(k)))
        t += ["", "**Market-neutral target (innovation), 87 features, 3-seed ensemble**", "", md(rows)]
    return "\n".join(t)


def md_model(m):
    lines = ["| Inputs / target | DA % | Peer-DA % | IC | MAPE % | L/S Sharpe | seeds |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fs in ["base", "full", "xs_base", "xs_full"]:
        r = summ(m, fs)
        if not r:
            continue
        cell = lambda k, d: f(r["mean"].get(k), d) + ("" if not r["std"].get(k) else f" ± {f(r['std'][k], d)}")
        lines.append(f"| {FSN[fs]} | {cell('DA_%', 2)} | {cell('DA_xs_%', 2)} | {cell('IC', 4)} | {cell('MAPE_%', 3)} | "
                     f"{cell('Sharpe_LS', 2)} | {r.get('n_seeds', '')} |")
    return "\n".join(lines)


def fill(path, txt):
    if not os.path.exists(path):
        return
    s = open(path).read()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        s = s[:s.index(a) + len(a)] + "\n" + txt + "\n" + s[s.index(b):]
        open(path, "w").write(s)
        print("   filled", os.path.relpath(path, ROOT))


fill(os.path.join(ROOT, "README.md"), md_main())
for m, d in [("lstm", "LSTM"), ("transformer", "Transformer"), ("informer", "Informer")]:
    fill(os.path.join(ROOT, d, "README.md"), md_model(m))
