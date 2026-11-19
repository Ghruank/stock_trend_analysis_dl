import { useEffect, useMemo, useState } from "react";
import {
  Area, Bar, BarChart, ComposedChart, Brush, CartesianGrid, Cell, ErrorBar, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { Moon, Sun } from "lucide-react";

/* ------------------------------------------------------------------ types & constants */
type Dict<T = any> = Record<string, T>;
const MODELS = ["lstm", "transformer", "informer"] as const;
type ModelKey = (typeof MODELS)[number];
const NAME: Dict<string> = { lstm: "LSTM", transformer: "Transformer", informer: "Informer", ridge: "Ridge (linear)", naive: "Naive random walk", combo: "3-model ensemble" };
const COLOR: Dict<string> = { lstm: "var(--s1)", transformer: "var(--s2)", informer: "var(--s3)", ridge: "var(--base)", naive: "var(--base)", combo: "var(--s4)" };
const FS_NAME: Dict<string> = { base: "Base (8 price/volume features)", full: "Full (87 engineered)", xs_full: "Full + market-neutral target", xs_base: "Base + market-neutral target" };

const METRICS: Dict<{ label: string; better: "high" | "low"; fmt: (v: number) => string; help: string }> = {
  "DA_xs_%": { label: "Beat-the-peers accuracy", better: "high", fmt: (v) => v.toFixed(2) + "%", help: "Share of asset-days where the model correctly says whether the asset will beat or lag the average of its regional peers tomorrow. 50% = coin flip." },
  "DA_%": { label: "Directional accuracy (1-day)", better: "high", fmt: (v) => v.toFixed(2) + "%", help: "Share of days where the sign of the predicted next-day return matches reality. 50% = coin flip." },
  IC: { label: "Information coefficient", better: "high", fmt: (v) => v.toFixed(4), help: "Mean daily cross-sectional Spearman rank correlation between forecasts and realised returns (within region). Quant funds consider 0.02-0.05 useful." },
  "R2_oos_%": { label: "Out-of-sample R² on returns", better: "high", fmt: (v) => v.toFixed(3) + "%", help: "1 - SSE(model)/SSE(zero forecast). > 0 means the model beats the random walk in squared error." },
  "MAPE_%": { label: "Next-day price MAPE", better: "low", fmt: (v) => v.toFixed(3) + "%", help: "Mean absolute % error of the predicted next-day close. Compare with the naive forecast (today's close)." },
  Sharpe_LS: { label: "Long/short Sharpe (gross)", better: "high", fmt: (v) => v.toFixed(2), help: "Annualised Sharpe of buying the top-20% and shorting the bottom-20% forecasts each day, within region, before costs." },
  Sharpe_dir: { label: "Directional Sharpe (gross)", better: "high", fmt: (v) => v.toFixed(2), help: "Equal-weight portfolio that is long when the model predicts up and short when it predicts down, before costs." },
  Sharpe_dir_net: { label: "Directional Sharpe (10 bps cost)", better: "high", fmt: (v) => v.toFixed(2), help: "Same as above after 10 bps per position change." },
  "DA_5d_%": { label: "Directional accuracy (5-day)", better: "high", fmt: (v) => v.toFixed(2) + "%", help: "Sign accuracy of the cumulative 5-day forecast." },
  "R2_oos_5d_%": { label: "Out-of-sample R² (5-day)", better: "high", fmt: (v) => v.toFixed(3) + "%", help: "R²_oos on cumulative 5-day returns." },
  R2_price: { label: "R² on price level", better: "high", fmt: (v) => v.toFixed(5), help: "≈1 for every model including the naive forecast, because tomorrow's price ≈ today's — shown only to illustrate why price-level R² is not informative." },
};

/* ------------------------------------------------------------------ helpers */
const useJSON = (url: string | null) => {
  const [d, setD] = useState<any>(null);
  useEffect(() => {
    if (!url) return;
    let ok = true;
    fetch(url).then((r) => r.json()).then((j) => ok && setD(j)).catch(() => ok && setD(undefined));
    return () => { ok = false; };
  }, [url]);
  return d;
};
const f2 = (v: number | null | undefined, k = 2) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(k));
const Legend = ({ items }: { items: { key: string; label: string; color: string; dash?: boolean }[] }) => (
  <div className="legend">
    {items.map((i) => (
      <span key={i.key}>
        <svg width="18" height="10"><line x1="1" x2="17" y1="5" y2="5" stroke={i.color} strokeWidth={i.dash ? 2 : 3} strokeDasharray={i.dash ? "4 3" : undefined} strokeLinecap="round" /></svg>
        {i.label}
      </span>
    ))}
  </div>
);
const Seg = ({ value, options, onChange }: { value: string; options: [string, string][]; onChange: (v: string) => void }) => (
  <div className="seg" role="tablist">
    {options.map(([k, l]) => (
      <button key={k} className={value === k ? "on" : ""} onClick={() => onChange(k)}>{l}</button>
    ))}
  </div>
);
const TT = ({ active, payload, label, fmt }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      {label != null && <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>}
      {payload.filter((p: any) => p.value != null).map((p: any) => (
        <div className="row" key={p.dataKey + p.name}>
          <span><span className="dot" style={{ background: p.color || p.payload?.fill }} /> {p.name}</span>
          <b>{fmt ? fmt(p.value) : typeof p.value === "number" ? p.value.toFixed(3) : p.value}</b>
        </div>
      ))}
    </div>
  );
};

/* ------------------------------------------------------------------ app */
const TABS: [string, string][] = [
  ["overview", "Overview"], ["results", "Results"], ["features", "Feature engineering"], ["breakdown", "Volatility & assets"],
  ["innovation", "Innovation"], ["explorer", "Prediction explorer"], ["backtest", "Backtest"], ["live", "Live demo"], ["training", "Training"],
];

export default function App() {
  const S = useJSON("data/summary.json");
  const [tab, setTab] = useState(() => (location.hash.slice(1) || "overview"));
  const [theme, setTheme] = useState<string | null>(null);
  useEffect(() => { location.hash = tab; window.scrollTo(0, 0); }, [tab]);
  useEffect(() => {
    const h = () => setTab(location.hash.slice(1) || "overview");
    window.addEventListener("hashchange", h);
    return () => window.removeEventListener("hashchange", h);
  }, []);
  useEffect(() => { if (theme) document.documentElement.setAttribute("data-theme", theme); }, [theme]);
  const dark = theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;

  return (
    <>
      <header className="top">
        <div className="wrap">
          <div className="brand">Stock Transformers<small>LSTM · Transformer · Informer</small></div>
          <nav className="tabs">
            {TABS.map(([k, l]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}
          </nav>
          <button className="theme-btn" aria-label="toggle theme" onClick={() => setTheme(dark ? "light" : "dark")}>{dark ? <Sun size={16} /> : <Moon size={16} />}</button>
        </div>
      </header>
      <main className="wrap">
        {S === null && <p className="lead" style={{ marginTop: 40 }}>Loading results…</p>}
        {S === undefined && <p className="lead" style={{ marginTop: 40 }}>Could not load data/summary.json — run scripts/export_web_data.py.</p>}
        {S && (
          <>
            {tab === "overview" && <Overview S={S} go={setTab} />}
            {tab === "results" && <Results S={S} />}
            {tab === "features" && <Features S={S} />}
            {tab === "breakdown" && <Breakdown S={S} />}
            {tab === "innovation" && <Innovation S={S} />}
            {tab === "explorer" && <Explorer S={S} />}
            {tab === "backtest" && <Backtest S={S} />}
            {tab === "live" && <Live S={S} />}
            {tab === "training" && <Training S={S} />}
          </>
        )}
        <footer>
          Stock Transformers — Khush Agrawal, Yashvi Mehta, Ghruank Kothare. Deep Learning project.
          Research only, not investment advice.
        </footer>
      </main>
    </>
  );
}

/* ------------------------------------------------------------------ helpers on data */
const summaryOf = (S: any, m: string, fs: string) => S.summary.find((r: any) => r.model === m && r.features === fs);
const metricOf = (S: any, m: string, fs: string, k: string, ens = true) => {
  const r = summaryOf(S, m, fs);
  if (!r) return null;
  return ens && r.ensemble ? r.ensemble[k] : r.mean[k];
};
const bestModel = (S: any, k = "IC", fs = "full") => {
  const xs = [...MODELS, "combo"].map((m) => ({ m, v: metricOf(S, m, fs, k) })).filter((x) => x.v != null);
  return xs.sort((a, b) => (METRICS[k].better === "high" ? b.v - a.v : a.v - b.v))[0];
};

/* ------------------------------------------------------------------ Overview */
function Overview({ S, go }: { S: any; go: (t: string) => void }) {
  const nA = S.meta.tickers.length;
  const hasXs = S.summary.some((r: any) => r.features === "xs_full");
  const fsX = hasXs ? "xs_full" : "full";
  const bIC = bestModel(S, "IC", fsX);
  const bDA = bestModel(S, "DA_%");
  const bLS = bestModel(S, "Sharpe_LS", fsX);
  const cov = S.cone?.coverage?.lstm;
  return (
    <>
      <h1>Can deep sequence models predict the stock market?</h1>
      <p className="lead">
        {nA} assets (Indian large & small caps, US mega-caps, commodities), daily data 2005 → Nov 2026,
        {" "}{S.meta.n_features.full} engineered features, and three architectures — LSTM, Transformer (Time2Vec) and Informer (ProbSparse attention) —
        compared on a strictly chronological, never-seen test period (Jan 2024 – Nov 2026).
      </p>
      <div className="grid g4">
        <div className="card kpi"><div className="l">Best next-day direction accuracy</div><div className="v">{bDA ? bDA.v.toFixed(2) + "%" : "—"}</div><div className="d">{bDA && NAME[bDA.m]} · coin flip = 50%</div></div>
        <div className="card kpi"><div className="l">Best information coefficient{hasXs ? " (market-neutral)" : ""}</div><div className="v">{bIC ? bIC.v.toFixed(3) : "—"}</div><div className="d">{bIC && NAME[bIC.m]} · raw-target LSTM {f2(metricOf(S, "lstm", "full", "IC"), 3)}</div></div>
        <div className="card kpi"><div className="l">Best long/short Sharpe (gross{hasXs ? ", market-neutral" : ""})</div><div className="v">{bLS ? bLS.v.toFixed(2) : "—"}</div><div className="d">{bLS && NAME[bLS.m]} · before costs</div></div>
        <div className="card kpi"><div className="l">Forecast range reliability</div><div className="v">{cov ? cov["80"].all.toFixed(1) + "%" : "—"}</div><div className="d">real price inside the 80% forecast cone · target 80%</div></div>
      </div>

      <h2>Pipeline</h2>
      <div className="flow">
        {[
          ["Frozen snapshot", "yfinance, 49 assets + 7 market series, 2005–2026"],
          ["Cleaning", "bad ticks dropped, demergers & futures rolls back-adjusted"],
          ["87 features", "momentum, volatility, oscillators, trend, liquidity, market, cross-section, calendar"],
          ["Train-only scaling", "per-asset z-score fitted 2007–2021, clipped ±5σ"],
          ["96-day windows", "target = next 5 daily returns / σ (stationary)"],
          ["LSTM · Transformer · Informer", "HP search on 2022–23 val, 3 seeds + ensemble"],
          ["Evaluation Jan 2024 – Nov 2026", "DA, IC, R²oos, MAPE vs naive, Sharpe, cone coverage"],
        ].map(([b, s], i, a) => (
          <div key={b} style={{ display: "contents" }}>
            <div className="step"><b>{b}</b><span>{s}</span></div>
            {i < a.length - 1 && <div className="arrow">→</div>}
          </div>
        ))}
      </div>

      {hasXs && (
        <>
          <h2>Headline: market-neutral target (our innovation), seed ensemble, test 2024–2026</h2>
          <HeadlineTable S={S} fs="xs_full" />
        </>
      )}
      <h2>Raw return target (full feature set, seed ensemble, test 2024–2026)</h2>
      <HeadlineTable S={S} />
      <div className="grid g3" style={{ marginTop: 16 }}>
        <div className="card"><h3>Forecast cones</h3><p className="sub">Every forecast comes with a calibrated 80% / 95% range for the next 5 days. Slide through the test period and see where the real price landed.</p><a href="#explorer" onClick={(e) => { e.preventDefault(); go("explorer"); }}>Prediction explorer →</a></div>
        <div className="card"><h3>How it works</h3><p className="sub">Stationary volatility-normalised targets, chronological splits with purging, 49-asset pooled training, 87 engineered features, honest baselines — and a market-neutral target that asks "which asset beats its peers?".</p><a href="#innovation" onClick={(e) => { e.preventDefault(); go("innovation"); }}>The innovation →</a></div>
        <div className="card"><h3>Real-life use</h3><p className="sub">Daily ranking signal for a market-neutral long/short book, or a risk-desk early warning of 5-day moves — run live on today's prices.</p><a href="#live" onClick={(e) => { e.preventDefault(); go("live"); }}>Live demo →</a></div>
      </div>
    </>
  );
}

function HeadlineTable({ S, fs = "full" }: { S: any; fs?: string }) {
  const cols = ["DA_%", "DA_xs_%", "IC", "R2_oos_%", "MAPE_%", "Sharpe_LS", "Sharpe_dir_net", "DA_5d_%"];
  const xs = fs.startsWith("xs");
  const rows = [
    ...[...MODELS, ...(summaryOf(S, "combo", fs) ? ["combo"] : [])].map((m) => ({ k: m, name: NAME[m], v: (c: string) => metricOf(S, m, fs, c) })),
    { k: "ridge", name: NAME.ridge, v: (c: string) => (xs ? S.baselines.ridge_xs : S.baselines.ridge)?.[c] },
    { k: "naive", name: NAME.naive, v: (c: string) => S.baselines.naive?.[c] },
  ];
  const best: Dict<number> = {};
  cols.forEach((c) => {
    const vals = rows.map((r) => r.v(c)).filter((x) => x != null && !Number.isNaN(x) && x !== 0 && !(c === "DA_xs_%" && x === 50));
    best[c] = METRICS[c].better === "high" ? Math.max(...vals) : Math.min(...vals);
  });
  return (
    <div className="card scroll">
      <table>
        <thead><tr><th>Model</th>{cols.map((c) => <th key={c} title={METRICS[c].help}>{METRICS[c].label}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.k}>
              <td><span className="dot" style={{ background: COLOR[r.k], marginRight: 6 }} />{r.name}</td>
              {cols.map((c) => {
                const v = r.v(c);
                return <td key={c} style={{ fontWeight: v === best[c] ? 700 : 400 }}>{v == null || Number.isNaN(v) ? "—" : METRICS[c].fmt(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="sub" style={{ marginTop: 6 }}>Bold = best in column. Hover a header for the definition. Naive = predict "tomorrow's close = today's close".</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Results */
function Results({ S }: { S: any }) {
  const [k, setK] = useState("IC");
  const [fs, setFs] = useState("full");
  const [mode, setMode] = useState("seeds");
  const data = [
    ...[...MODELS, "combo"].map((m) => {
      const r = summaryOf(S, m, fs);
      const v = r ? (mode === "ens" && r.ensemble ? r.ensemble[k] : r.mean[k]) : null;
      return { key: m, name: NAME[m], v, err: mode === "seeds" && r ? r.std[k] : 0 };
    }),
    ...(fs === "full" ? [{ key: "ridge", name: "Ridge", v: S.baselines.ridge?.[k], err: 0 }] : []),
    ...(fs === "xs_full" ? [{ key: "ridge", name: "Ridge", v: S.baselines.ridge_xs?.[k], err: 0 }] : []),
    ...(S.baselines.naive?.[k] != null && !["DA_%", "DA_5d_%"].includes(k) ? [{ key: "naive", name: "Naive", v: S.baselines.naive[k], err: 0 }] : []),
  ].filter((d) => d.v != null);
  const ref = k.startsWith("DA") ? 50 : 0;
  return (
    <>
      <h1>Model comparison</h1>
      <p className="lead">Every number is out-of-sample: models were tuned on 2022–2023 and evaluated once on Jan 2024 – Nov 2026. Error bars show ±1 s.d. across 3 random seeds.</p>
      <div className="controls">
        <select value={k} onChange={(e) => setK(e.target.value)}>
          {Object.entries(METRICS).map(([kk, m]) => <option key={kk} value={kk}>{m.label}</option>)}
        </select>
        <Seg value={fs} onChange={setFs} options={[["xs_full", "Market-neutral target"], ["full", "Full features"], ["base", "Base features"]].filter(([f]) => S.summary.some((r: any) => r.features === f)) as [string, string][]} />
        <Seg value={mode} onChange={setMode} options={[["seeds", "Mean of seeds"], ["ens", "Seed ensemble"]]} />
      </div>
      <div className="note">{METRICS[k].help} {METRICS[k].better === "low" ? "Lower is better." : "Higher is better."}</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 16, right: 16, left: 8, bottom: 4 }} barCategoryGap="30%">
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} />
            <YAxis tickLine={false} axisLine={false} domain={k.startsWith("DA") ? [48, "auto"] : ["auto", "auto"]} tickFormatter={(v) => (Math.abs(v) < 0.1 && v !== 0 ? v.toFixed(3) : v.toFixed(2))} />
            <ReferenceLine y={ref} stroke="var(--axis)" />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={METRICS[k].fmt} />} />
            <Bar dataKey="v" name={METRICS[k].label} radius={[4, 4, 0, 0]} maxBarSize={64}>
              {data.map((d) => <Cell key={d.key} fill={COLOR[d.key]} />)}
              <ErrorBar dataKey="err" width={6} stroke="var(--ink-2)" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <h2>All metrics — {FS_NAME[fs]}</h2>
      <HeadlineTable S={S} fs={fs} />
      <h2>Base → Full: what feature engineering bought each model</h2>
      <FsGain S={S} />
    </>
  );
}

function FsGain({ S }: { S: any }) {
  const ks = ["DA_%", "IC", "R2_oos_%", "Sharpe_LS"];
  const [k, setK] = useState("IC");
  const data = MODELS.map((m) => ({ name: NAME[m], key: m, base: metricOf(S, m, "base", k, false), full: metricOf(S, m, "full", k, false) }));
  return (
    <div className="card">
      <div className="controls"><Seg value={k} onChange={setK} options={ks.map((x) => [x, METRICS[x].label.replace(" (1-day)", "")])} /></div>
      <Legend items={[{ key: "b", label: "Base (8 features)", color: "var(--axis)" }, { key: "f", label: "Full (87 features)", color: "var(--ink-2)" }]} />
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 4 }} barGap={2}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="name" tickLine={false} />
          <YAxis tickLine={false} axisLine={false} domain={k.startsWith("DA") ? [48, "auto"] : ["auto", "auto"]} />
          <ReferenceLine y={k.startsWith("DA") ? 50 : 0} stroke="var(--axis)" />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={METRICS[k].fmt} />} />
          <Bar dataKey="base" name="Base" radius={[4, 4, 0, 0]} maxBarSize={44}>{data.map((d) => <Cell key={d.key} fill={COLOR[d.key]} fillOpacity={0.4} />)}</Bar>
          <Bar dataKey="full" name="Full" radius={[4, 4, 0, 0]} maxBarSize={44}>{data.map((d) => <Cell key={d.key} fill={COLOR[d.key]} />)}</Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="sub">Lighter bar = base feature set, solid = full. Mean over 3 seeds.</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Features */
const GROUP_DESC: Dict<string> = {
  base: "8 basic price/volume features, made stationary: return, OHLC relative to close, volume z-score, ROC5, RSI5, Bollinger width",
  momentum: "log-returns over 2–250 days",
  candle: "gap, intraday body, range, close-location value, shadows",
  volatility: "realised vol 5/20/60, Parkinson, Garman–Klass, ATR, vol-of-vol, skew, kurtosis, autocorrelation",
  oscillator: "RSI(2,14) Wilder, MACD & histogram, stochastic %K/%D, CCI, MFI, ADX, DI spread",
  trend: "price vs SMA 10/20/50/200, Bollinger %B & width, 52-week high/low distance, fractionally-differenced log price (d=0.4)",
  volume_liquidity: "volume ratio, Δlog volume, OBV slope, Amihud illiquidity, turnover z-score",
  market: "home index returns/vol/trend, VIX level & change, lagged S&P/NIFTY, crude, FX; rolling beta, residual return, relative strength, correlation",
  cross_sectional: "same-day percentile ranks across the region (1/5/20/60-day returns, vol, volume), demeaned return",
  calendar: "day-of-week, month, day-of-month (sin/cos)",
};
function Features({ S }: { S: any }) {
  const groups = useMemo(() => {
    const g: Dict<string[]> = {};
    S.meta.features.forEach((f: any) => { (g[f.group] ||= []).push(f.name); });
    return g;
  }, [S]);
  const A = S.ablation;
  const [k, setK] = useState("delta_val_loss");
  const abl = A ? A.groups.map((g: any) => ({
    group: g.group.replace("_", " "), n: g.n_features_dropped,
    v: k === "delta_val_loss" ? g.delta_val_loss * 1000 : g.test[k] - A.ref_test[k],
  })).sort((a: any, b: any) => b.v - a.v) : [];
  return (
    <>
      <h1>Feature engineering</h1>
      <p className="lead">Instead of feeding raw prices, we engineered {S.meta.n_features.full} stationary features in 10 families — every one computable from information available at the close of day t (US/commodity series are lagged a day for Indian assets).</p>
      <div className="grid g2">
        {Object.entries(groups).map(([g, names]) => (
          <div className="card" key={g}>
            <h3>{g.replace("_", " ")} <span className="sub">· {names.length}</span></h3>
            <div className="sub" style={{ marginBottom: 6 }}>{GROUP_DESC[g]}</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {names.map((n) => <code key={n} style={{ fontSize: 11.5, background: "var(--surface-2)", padding: "1px 6px", borderRadius: 5 }}>{n.slice(2)}</code>)}
            </div>
          </div>
        ))}
      </div>
      <h2>Ablation: remove one feature family at a time</h2>
      {A ? (
        <>
          <p className="lead">Model: {NAME[A.model]} (full config), 2 seeds per ablation. A family is valuable if removing it <b>raises</b> validation loss or <b>lowers</b> test skill.</p>
          <div className="controls"><Seg value={k} onChange={setK} options={[["delta_val_loss", "Δ validation loss (×1000)"], ["IC", "Δ test IC"], ["DA_%", "Δ test DA"], ["Sharpe_LS", "Δ test L/S Sharpe"]]} /></div>
          <div className="card">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={abl} layout="vertical" margin={{ top: 8, right: 24, left: 24, bottom: 4 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} tickFormatter={(v) => v.toFixed(k === "IC" ? 3 : 2)} />
                <YAxis type="category" dataKey="group" width={120} tickLine={false} axisLine={false} />
                <ReferenceLine x={0} stroke="var(--axis)" />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={(v: number) => v.toFixed(4)} />} />
                <Bar dataKey="v" name="change when removed" radius={[0, 4, 4, 0]} maxBarSize={22}>
                  {abl.map((d: any) => {
                    const helpful = k === "delta_val_loss" ? d.v > 0 : d.v < 0;
                    return <Cell key={d.group} fill={helpful ? "var(--s1)" : "var(--axis)"} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="sub">Blue = the model got worse without this family (the family helps). Grey = no measurable help.</div>
          </div>
        </>
      ) : <div className="note">Ablation results not available yet.</div>}
    </>
  );
}

/* ------------------------------------------------------------------ Breakdown */
function Breakdown({ S }: { S: any }) {
  const [by, setBy] = useState("vol_tier");
  const [k, setK] = useState("DA_%");
  const order = by === "vol_tier" ? ["low", "mid", "high"] : ["equity_in", "equity_us", "commodity"];
  const label: Dict<string> = { low: "Low volatility", mid: "Mid volatility", high: "High volatility", equity_in: "Indian equities", equity_us: "US equities", commodity: "Commodities" };
  const data = order.map((g) => {
    const row: Dict = { g: label[g] };
    MODELS.forEach((m) => { row[m] = S.breakdown[`${m}:full`]?.[by]?.[g]?.[k]; });
    row.ridge = S.baseline_breakdown?.ridge?.[by]?.[g]?.[k];
    if (k === "MAPE_%") row.naive = S.breakdown[`lstm:full`]?.[by]?.[g]?.["MAPE_naive_%"];
    return row;
  });
  const members = (g: string) => S.series_index.filter((x: any) => x[by] === g).map((x: any) => x.ticker);
  const series = [...MODELS, "ridge", ...(k === "MAPE_%" ? ["naive"] : [])];
  return (
    <>
      <h1>Volatile vs non-volatile, shares vs commodities</h1>
      <p className="lead">Assets are tiered by their 2007–2021 annualised volatility (tertiles) and by asset class.</p>
      <div className="controls">
        <Seg value={by} onChange={setBy} options={[["vol_tier", "Volatility tier"], ["asset_class", "Asset class"]]} />
        <Seg value={k} onChange={setK} options={[["DA_%", "Directional accuracy"], ["R2_oos_%", "R² oos"], ["MAPE_%", "Price MAPE"]]} />
      </div>
      <div className="card">
        <Legend items={series.map((m) => ({ key: m, label: NAME[m], color: COLOR[m], dash: m === "naive" }))} />
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 4 }} barGap={2}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="g" tickLine={false} />
            <YAxis tickLine={false} axisLine={false} domain={k === "DA_%" ? [46, "auto"] : ["auto", "auto"]} />
            <ReferenceLine y={k === "DA_%" ? 50 : 0} stroke="var(--axis)" />
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={METRICS[k].fmt} />} />
            {series.map((m) => <Bar key={m} dataKey={m} name={NAME[m]} fill={COLOR[m]} fillOpacity={m === "naive" ? 0.35 : 1} radius={[4, 4, 0, 0]} maxBarSize={36} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="grid g3" style={{ marginTop: 14 }}>
        {order.map((g) => <div className="card" key={g}><h3>{label[g]}</h3><div className="sub">{members(g).join(", ")}</div></div>)}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ Explorer */
function Explorer({ S }: { S: any }) {
  const [t, setT] = useState("TCS");
  const [on, setOn] = useState<Dict<boolean>>({ lstm: true, transformer: true, informer: true });
  const d = useJSON(`data/series/${encodeURIComponent(t)}.json`);
  const rows = useMemo(() => d ? d.dates.map((x: string, i: number) => {
    const r: Dict = { date: x, actual: d.actual[i] };
    MODELS.forEach((m) => { if (d.pred[m]) r[m] = d.pred[m][i]; });
    return r;
  }) : [], [d]);
  const sc = useMemo(() => d ? MODELS.filter((m) => d.pred_ret[m]).map((m) => ({ m, pts: d.pred_ret[m].map((p: number, i: number) => ({ x: p, y: d.actual_ret[i] })).filter((q: any) => q.x != null) })) : [], [d]);
  const stats = useMemo(() => {
    if (!d) return [];
    return MODELS.filter((m) => d.pred_ret[m]).map((m) => {
      const p = d.pred_ret[m], a = d.actual_ret;
      let hit = 0, n = 0, ae = 0, aen = 0, c = 0;
      for (let i = 0; i < a.length; i++) {
        if (p[i] == null || d.pred[m][i] == null || a[i] == null) continue;
        c++;
        if (a[i] !== 0) { n++; if (Math.sign(p[i]) === Math.sign(a[i])) hit++; }
        ae += Math.abs(d.pred[m][i] - d.actual[i]) / d.actual[i];
        aen += Math.abs(d.prev[i] - d.actual[i]) / d.actual[i];
      }
      return { m, da: (hit / Math.max(n, 1)) * 100, mape: (ae / Math.max(c, 1)) * 100, naive: (aen / Math.max(c, 1)) * 100 };
    });
  }, [d]);
  const info = S.series_index.find((x: any) => x.ticker === t);
  return (
    <>
      <h1>Prediction explorer</h1>
      <p className="lead">Pick any of the 49 assets. First: a 5-day forecast with its predicted range, and where the price really went. Below: the next-day close predicted by each model vs the real close over the whole Jan 2024 – Nov 2026 test period.</p>
      <div className="controls">
        <select value={t} onChange={(e) => setT(e.target.value)}>
          {["equity_in", "equity_us", "commodity"].map((g) => (
            <optgroup key={g} label={g}>{S.series_index.filter((x: any) => x.asset_class === g).map((x: any) => <option key={x.ticker} value={x.ticker}>{x.ticker}</option>)}</optgroup>
          ))}
        </select>
        {MODELS.map((m) => (
          <span key={m} className={"chip" + (on[m] ? " on" : "")} onClick={() => setOn({ ...on, [m]: !on[m] })}>
            <span className="dot" style={{ background: on[m] ? COLOR[m] : "var(--axis)" }} />{NAME[m]}
          </span>
        ))}
        {info && <span className="sub">{info.asset_class} · {info.vol_tier} volatility</span>}
      </div>
      {d && S.cone && <ConeExplorer S={S} d={d} t={t} />}
      <h2>Next-day close, every day of the test period</h2>
      <div className="card">
        <Legend items={[{ key: "a", label: "Actual close", color: "var(--ink)" }, ...MODELS.filter((m) => on[m]).map((m) => ({ key: m, label: NAME[m] + " forecast", color: COLOR[m] }))]} />
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} minTickGap={48} />
            <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} width={64} />
            <Tooltip content={<TT fmt={(v: number) => v.toFixed(2)} />} />
            <Line dataKey="actual" name="Actual" stroke="var(--ink)" dot={false} strokeWidth={2} isAnimationActive={false} />
            {MODELS.filter((m) => on[m]).map((m) => <Line key={m} dataKey={m} name={NAME[m]} stroke={COLOR[m]} dot={false} strokeWidth={1.5} isAnimationActive={false} />)}
            <Brush dataKey="date" height={22} stroke="var(--axis)" fill="var(--surface)" travellerWidth={8} startIndex={Math.max(0, rows.length - 120)} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3>Per-asset scorecard ({t})</h3>
          <table>
            <thead><tr><th>Model</th><th>Direction hit-rate</th><th>Price MAPE</th><th>Naive MAPE</th></tr></thead>
            <tbody>{stats.map((s) => <tr key={s.m}><td><span className="dot" style={{ background: COLOR[s.m], marginRight: 6 }} />{NAME[s.m]}</td><td>{s.da.toFixed(2)}%</td><td>{s.mape.toFixed(3)}%</td><td>{s.naive.toFixed(3)}%</td></tr>)}</tbody>
          </table>
          <div className="sub" style={{ marginTop: 6 }}>Price lines overlap almost perfectly because tomorrow's price ≈ today's — the edge lives in the <i>direction</i> and size of the move, shown on the right.</div>
        </div>
        <div className="card">
          <h3>Predicted vs realised next-day return (%)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid />
              <XAxis type="number" dataKey="x" name="predicted" tickLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} />
              <YAxis type="number" dataKey="y" name="actual" tickLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
              <ZAxis range={[10, 10]} />
              <ReferenceLine x={0} stroke="var(--axis)" /><ReferenceLine y={0} stroke="var(--axis)" />
              <Tooltip content={<TT fmt={(v: number) => v.toFixed(2) + "%"} />} />
              {sc.filter((s) => on[s.m]).map((s) => <Scatter key={s.m} name={NAME[s.m]} data={s.pts} fill={COLOR[s.m]} fillOpacity={0.45} isAnimationActive={false} />)}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}


/* ------------------------------------------------------------------ Forecast cone */
type ConeQ = { lo: number[]; hi: number[] };
const coneBand = (c0: number, cum: number[], sig: number, q: ConeQ) => ({
  lo: [c0, ...cum.map((v, k) => c0 * Math.exp(v + q.lo[k] * sig))],
  hi: [c0, ...cum.map((v, k) => c0 * Math.exp(v + q.hi[k] * sig))],
});

function ConeExplorer({ S, d, t }: { S: any; d: any; t: string }) {
  const [m, setM] = useState<string>("lstm");
  const [view, setView] = useState("single");
  const n = d?.d0?.length ?? 0;
  const [i, setI] = useState(-1);
  useEffect(() => setI(-1), [t]);
  const idx = i < 0 ? Math.max(0, n - 6) : Math.min(i, n - 6);
  const Q80: ConeQ = S.cone.levels["80"], Q95: ConeQ = S.cone.levels["95"];
  const cov = S.cone.coverage[m];
  const single = useMemo(() => {
    if (!d?.cum5?.[m] || n < 7) return null;
    const c0 = d.prev[idx], sg = d.sigma[idx], cum = d.cum5[m][idx];
    const b80 = coneBand(c0, cum, sg, Q80), b95 = coneBand(c0, cum, sg, Q95);
    const rows: Dict[] = [];
    for (let j = Math.max(0, idx - 40); j < idx; j++) rows.push({ date: d.d0[j], hist: d.prev[j] });
    const act = [c0, ...d.path[idx]];
    for (let k = 0; k <= 5; k++) {
      rows.push({ date: d.d0[idx + k], hist: k === 0 ? c0 : undefined, act: act[k], pred: c0 * Math.exp(k ? cum[k - 1] : 0),
        b80: [b80.lo[k], b80.hi[k]], b95: [b95.lo[k], b95.hi[k]] });
    }
    const end = act[5];
    const where = end >= b80.lo[5] && end <= b80.hi[5] ? "inside the 80% band" : end >= b95.lo[5] && end <= b95.hi[5] ? "inside the 95% band (outside 80%)" : "outside both bands — a surprise move";
    return { rows, where, ret: (Math.exp(cum[4]) - 1) * 100, real: (end / c0 - 1) * 100, w80: (b80.hi[5] / b80.lo[5] - 1) * 100 };
  }, [d, m, idx, n]);
  const chain = useMemo(() => {
    if (!d?.cum5?.[m] || n < 30) return [];
    const start = Math.max(0, n - 6 - 125);
    const rows: Dict[] = [];
    const at = (date: string) => { let r = rows.find((x) => x.date === date); if (!r) { r = { date }; rows.push(r); } return r; };
    for (let j = start; j < n; j++) at(d.d0[j]).hist = d.prev[j];
    let par = 0;
    for (let j = start; j + 6 <= n; j += 5, par ^= 1) {
      const c0 = d.prev[j], sg = d.sigma[j], cum = d.cum5[m][j];
      const b80 = coneBand(c0, cum, sg, Q80), b95 = coneBand(c0, cum, sg, Q95);
      for (let k = 0; k <= 5; k++) {
        const r = at(d.d0[j + k]);
        r[`b80_${par}`] = [b80.lo[k], b80.hi[k]]; r[`b95_${par}`] = [b95.lo[k], b95.hi[k]];
        r[`p_${par}`] = c0 * Math.exp(k ? cum[k - 1] : 0);
      }
    }
    return rows;
  }, [d, m, n]);
  const fmtB = (v: any) => Array.isArray(v) ? `${v[0].toFixed(2)} – ${v[1].toFixed(2)}` : v.toFixed(2);
  return (
    <>
      <h2>Forecast cone: the predicted range vs what actually happened</h2>
      <p className="lead">Each forecast is a 5-day path (dashed) plus a calibrated range: the band width comes from how far prices moved in 2022–23 (validation), scaled by the asset's current volatility. Over the whole test set and all 49 assets, the real price ended up inside the {m.toUpperCase() === "LSTM" ? "LSTM's" : NAME[m] + "'s"} 80% band <b>{cov["80"].all.toFixed(1)}%</b> of the time and inside the 95% band <b>{cov["95"].all.toFixed(1)}%</b> — the ranges are honest.</p>
      <div className="controls">
        <Seg value={m} onChange={setM} options={MODELS.map((x) => [x, NAME[x]]) as [string, string][]} />
        <Seg value={view} onChange={setView} options={[["single", "One forecast"], ["chain", "Weekly cones, last 6 months"]]} />
      </div>
      {view === "single" && single && (
        <>
          <div className="controls">
            <button className="chip" onClick={() => setI(Math.max(0, idx - 5))}>◀ week</button>
            <input type="range" min={0} max={n - 6} value={idx} onChange={(e) => setI(+e.target.value)} style={{ flex: 1, minWidth: 160 }} aria-label="forecast date" />
            <button className="chip" onClick={() => setI(Math.min(n - 6, idx + 5))}>week ▶</button>
            <span className="sub">forecast made after the close of <b>{d.d0[idx]}</b></span>
          </div>
          <div className="grid g3">
            <div className="card kpi"><div className="l">Predicted 5-day move</div><div className="v">{(single.ret > 0 ? "+" : "") + single.ret.toFixed(2)}%</div><div className="d">80% range ±{(single.w80 / 2).toFixed(1)}%</div></div>
            <div className="card kpi"><div className="l">What actually happened</div><div className="v"><span className={single.real >= 0 ? "tag-up" : "tag-down"}>{(single.real > 0 ? "+" : "") + single.real.toFixed(2)}%</span></div><div className="d">{single.where}</div></div>
            <div className="card kpi"><div className="l">Direction call</div><div className="v">{Math.sign(single.ret) === Math.sign(single.real) ? "✓ right" : "✗ wrong"}</div><div className="d">the range matters more than the point forecast</div></div>
          </div>
          <div className="card" style={{ marginTop: 14 }}>
            <Legend items={[{ key: "h", label: "Price before the forecast", color: "var(--ink)" }, { key: "p", label: NAME[m] + " forecast path", color: COLOR[m], dash: true },
              { key: "b", label: "80% / 95% range", color: COLOR[m] }, { key: "a", label: "What actually happened", color: "var(--bad)" }]} />
            <ResponsiveContainer width="100%" height={360}>
              <ComposedChart data={single.rows} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="date" tickLine={false} minTickGap={40} />
                <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} width={64} tickFormatter={(v) => v.toFixed(0)} />
                <Tooltip content={<TT fmt={fmtB} />} />
                <Area dataKey="b95" name="95% range" stroke="none" fill={COLOR[m]} fillOpacity={0.14} isAnimationActive={false} />
                <Area dataKey="b80" name="80% range" stroke="none" fill={COLOR[m]} fillOpacity={0.3} isAnimationActive={false} />
                <ReferenceLine x={d.d0[idx]} stroke="var(--axis)" strokeDasharray="3 3" />
                <Line dataKey="hist" name="Close" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line dataKey="pred" name="Forecast" stroke={COLOR[m]} strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line dataKey="act" name="Actual" stroke="var(--bad)" strokeWidth={2.2} dot={{ r: 3 }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
      {view === "chain" && (
        <div className="card">
          <Legend items={[{ key: "h", label: "Actual close", color: "var(--ink)" }, { key: "p", label: NAME[m] + " forecast path", color: COLOR[m], dash: true }, { key: "b", label: "80% / 95% range", color: COLOR[m] }]} />
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={chain} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="date" tickLine={false} minTickGap={48} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} width={64} tickFormatter={(v) => v.toFixed(0)} />
              <Tooltip content={<TT fmt={fmtB} />} />
              {[0, 1].map((q) => <Area key={"95" + q} dataKey={`b95_${q}`} name="95% range" stroke="none" fill={COLOR[m]} fillOpacity={0.14} connectNulls={false} isAnimationActive={false} />)}
              {[0, 1].map((q) => <Area key={"80" + q} dataKey={`b80_${q}`} name="80% range" stroke="none" fill={COLOR[m]} fillOpacity={0.3} connectNulls={false} isAnimationActive={false} />)}
              {[0, 1].map((q) => <Line key={"p" + q} dataKey={`p_${q}`} name="Forecast" stroke={COLOR[m]} strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls={false} isAnimationActive={false} />)}
              <Line dataKey="hist" name="Close" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="sub">A new cone starts every 5 trading days from the actual close. The black line should stay inside the dark band about 4 weeks in 5.</div>
        </div>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ Backtest */
function Backtest({ S }: { S: any }) {
  const [kind, setKind] = useState("ls");
  const E = S.equity;
  const hasXs = Object.keys(E).some((k) => k.includes("_xs_"));
  const [tgt, setTgt] = useState(hasXs ? "xs" : "raw");
  const sfx = tgt === "xs" ? `_xs_${kind}` : `_${kind}`;
  const keys = [...MODELS, "combo", "ridge"].filter((m) => E[`${m}${sfx}`]);
  const rows = E.dates.map((d: string, i: number) => {
    const r: Dict = { date: d };
    keys.forEach((m) => { r[m] = E[`${m}${sfx}`][i]; });
    if (kind === "dir") r.bh = E.buy_hold[i];
    return r;
  });
  return (
    <>
      <h1>Does the signal make money? (paper backtest)</h1>
      <p className="lead">Cumulative log-return (%) of simple daily strategies built from each model's test-period forecasts. No leverage, no costs in the curves — see the cost-adjusted Sharpe in Results. Illustrative, not investment advice.</p>
      <div className="controls">
        <Seg value={kind} onChange={setKind} options={[["ls", "Long/short top vs bottom 20%"], ["dir", "Long-or-short each asset by sign"]]} />
        {hasXs && <Seg value={tgt} onChange={setTgt} options={[["xs", "Market-neutral models"], ["raw", "Raw-return models"]]} />}
      </div>
      <div className="card">
        <Legend items={[...keys.map((m) => ({ key: m, label: NAME[m], color: COLOR[m], dash: m === "ridge" })), ...(kind === "dir" ? [{ key: "bh", label: "Equal-weight buy & hold", color: "var(--ink)", dash: true }] : [])]} />
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} minTickGap={60} />
            <YAxis tickLine={false} axisLine={false} tickFormatter={(v) => v + "%"} />
            <ReferenceLine y={0} stroke="var(--axis)" />
            <Tooltip content={<TT fmt={(v: number) => v.toFixed(2) + "%"} />} />
            {keys.map((m) => <Line key={m} dataKey={m} name={NAME[m]} stroke={COLOR[m]} dot={false} strokeWidth={2} strokeDasharray={m === "ridge" ? "4 3" : undefined} isAnimationActive={false} />)}
            {kind === "dir" && <Line dataKey="bh" name="Buy & hold" stroke="var(--ink)" strokeDasharray="4 3" dot={false} strokeWidth={1.5} isAnimationActive={false} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="note">The long/short book is market-neutral, so it isolates the model's ranking skill (IC) from the market's direction. Daily rebalancing at 10 bps would cost roughly 25% a year in turnover — real deployment would need slower signals or lower costs.</div>
    </>
  );
}

/* ------------------------------------------------------------------ Innovation */
function Innovation({ S }: { S: any }) {
  const [k, setK] = useState("IC");
  const ms = [...MODELS, ...(summaryOf(S, "combo", "xs_full") ? ["combo"] : [])];
  const data = ms.map((m) => ({ key: m, name: NAME[m], raw: m === "combo" ? null : metricOf(S, m, "full", k), xs: metricOf(S, m, "xs_full", k) }));
  const R = S.ridge_ablation;
  const abl = R ? R.groups.map((g: any) => ({ name: g.n > 60 ? "only the 8 base features" : "drop " + g.group.replace("_", " ") + ` (${g.n})`, d: (g.IC - R.ref.IC) * 1000 }))
    .sort((a: any, b: any) => a.d - b.d) : [];
  const rx = S.baselines.ridge_xs?.[k];
  return (
    <>
      <h1>What is new: market-neutral targets + 87 engineered features</h1>
      <p className="lead">
        Predicting <i>where the whole market goes</i> tomorrow is almost pure noise — every model lands at a coin flip. But predicting
        <i> which asset will beat its peers</i> removes the market factor (most of the day-to-day variance) and leaves a learnable signal.
        We train every architecture on returns demeaned within (region, day) and choose checkpoints by validation correlation. Result: the LSTM learns a stable ranking signal, the Transformer a weaker one, the Informer none — and averaging all three is worse than the LSTM alone. A linear ridge on the same features is still the best ranker, so the features carry most of the signal.
      </p>
      <div className="flow">
        {[["Raw target", "r(t+1..t+5) / σ"], ["Subtract peer mean", "same region, same day"], ["Train LSTM · Transformer · Informer", "early-stop on val. correlation"], ["3-seed ensemble", "average per architecture"], ["Daily ranking", "long top 20% · short bottom 20%"]].map(([b, s2], i, a) => (
          <div key={b} style={{ display: "contents" }}>
            <div className="step"><b>{b}</b><span>{s2}</span></div>
            {i < a.length - 1 && <div className="arrow">→</div>}
          </div>
        ))}
      </div>
      <div className="controls" style={{ marginTop: 16 }}>
        <Seg value={k} onChange={setK} options={[["IC", "Information coefficient"], ["DA_xs_%", "Beat-the-peers accuracy"], ["Sharpe_LS", "Long/short Sharpe"]]} />
      </div>
      <div className="note">{METRICS[k].help}</div>
      <div className="card">
        <Legend items={[{ key: "r", label: "Raw return target (faded)", color: "var(--base)" }, { key: "x", label: "Market-neutral target", color: "var(--ink)" }, ...(rx != null ? [{ key: "rx", label: "Ridge, market-neutral", color: "var(--base)", dash: true }] : [])]} />
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={data} margin={{ top: 16, right: 16, left: 8, bottom: 4 }} barCategoryGap="25%">
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} />
            <YAxis tickLine={false} axisLine={false} domain={k === "DA_xs_%" ? [49, (mx: number) => Math.max(mx, rx ?? 0) + 0.2] : ["auto", (mx: number) => Math.max(mx, rx ?? 0) * 1.1]} tickFormatter={(v) => (k === "IC" ? v.toFixed(3) : v.toFixed(1))} />
            <ReferenceLine y={k === "DA_xs_%" ? 50 : 0} stroke="var(--axis)" />
            {rx != null && <ReferenceLine y={rx} stroke="var(--base)" strokeDasharray="4 3" />}
            <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={METRICS[k].fmt} />} />
            <Bar dataKey="raw" name="Raw target" radius={[4, 4, 0, 0]} maxBarSize={40}>{data.map((d) => <Cell key={d.key} fill={COLOR[d.key]} fillOpacity={0.35} />)}</Bar>
            <Bar dataKey="xs" name="Market-neutral" radius={[4, 4, 0, 0]} maxBarSize={40}>{data.map((d) => <Cell key={d.key} fill={COLOR[d.key]} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {R && (
        <>
          <h2>Which feature families carry the signal?</h2>
          <p className="lead">Linear probe on the market-neutral target: remove one family, refit, measure the change in test IC (×1000). Full set IC = {R.ref.IC.toFixed(4)}; the 8 basic price/volume features alone reach far less.</p>
          <div className="card">
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={abl} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 4 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} />
                <YAxis type="category" dataKey="name" width={190} tickLine={false} axisLine={false} />
                <ReferenceLine x={0} stroke="var(--axis)" />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} content={<TT fmt={(v: number) => (v > 0 ? "+" : "") + v.toFixed(2)} />} />
                <Bar dataKey="d" name="Δ IC ×1000" radius={[0, 4, 4, 0]} maxBarSize={20}>{abl.map((a: any) => <Cell key={a.name} fill={a.d < 0 ? "var(--s1)" : "var(--base)"} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="note">Blue = removing the family hurts, so it carries signal. Market context, volatility, cross-sectional ranks and oscillators matter most — all added in 2026.</div>
        </>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ Live demo */
function Live({ S }: { S: any }) {
  const L = S.live;
  const [t, setT] = useState(L?.assets?.[0]?.ticker ?? "TCS");
  if (!L) return <><h1>Live demo</h1><div className="note">Run <code>python live_predict.py --refresh</code> then re-export to populate this page.</div></>;
  const a = L.assets.find((x: any) => x.ticker === t) ?? L.assets[0];
  const rows: Dict[] = a.history.map(([d, c]: [string, number]) => ({ date: d, actual: c }));
  const last = rows[rows.length - 1];
  MODELS.forEach((m) => { if (a.forecasts[m]) last[m] = a.last_close; });
  const lcum = a.forecasts.lstm?.daily_ret?.reduce((acc: number[], v: number) => [...acc, (acc[acc.length - 1] ?? 0) + v], []);
  const lb80 = lcum && S.cone ? coneBand(a.last_close, lcum, a.sigma, S.cone.levels["80"]) : null;
  const lb95 = lcum && S.cone ? coneBand(a.last_close, lcum, a.sigma, S.cone.levels["95"]) : null;
  if (lb80 && lb95) { last.b80 = [lb80.lo[0], lb80.hi[0]]; last.b95 = [lb95.lo[0], lb95.hi[0]]; }
  a.future_dates.forEach((d: string, i: number) => {
    const r: Dict = { date: d };
    MODELS.forEach((m) => { if (a.forecasts[m]) r[m] = a.forecasts[m].path[i]; });
    if (lb80 && lb95) { r.b80 = [lb80.lo[i + 1], lb80.hi[i + 1]]; r.b95 = [lb95.lo[i + 1], lb95.hi[i + 1]]; }
    rows.push(r);
  });
  const table = L.assets.map((x: any) => ({ t: x.ticker, ...Object.fromEntries(MODELS.map((m) => [m, x.forecasts[m]?.["ret_5d_%"]])) }));
  const consensus = (x: any) => {
    const v = MODELS.map((m) => x[m]).filter((y) => y != null);
    const up = v.filter((y) => y > 0).length;
    return up === v.length ? "UP" : up === 0 ? "DOWN" : "MIXED";
  };
  return (
    <>
      <h1>Live demo — next 5 trading days</h1>
      <p className="lead">Forecasts generated by <code>live_predict.py</code> from the latest prices (data up to <b>{L.generated_for}</b>), through exactly the training pipeline: clean → 87 features → train-period scaler → 96-day window → model.</p>
      <div className="note"><b>Real-life scenario:</b> a portfolio manager runs this after the 3:30 pm NSE close. Assets where all three models agree on direction go on a watch-list; the ranked 5-day forecasts tilt a market-neutral long/short book, and large predicted moves flag positions for the risk desk to hedge.</div>
      <div className="controls">
        <select value={t} onChange={(e) => setT(e.target.value)}>{L.assets.map((x: any) => <option key={x.ticker}>{x.ticker}</option>)}</select>
        <span className="sub">last close {a.last_close.toFixed(2)} on {a.last_date} · daily σ {(a.sigma * 100).toFixed(2)}%</span>
      </div>
      <div className="grid g3">
        {MODELS.filter((m) => a.forecasts[m]).map((m) => (
          <div className="card kpi" key={m}>
            <div className="l"><span className="dot" style={{ background: COLOR[m], marginRight: 6 }} />{NAME[m]}</div>
            <div className="v"><span className={a.forecasts[m].signal === "UP" ? "tag-up" : "tag-down"}>{a.forecasts[m].signal === "UP" ? "▲" : "▼"} {a.forecasts[m]["ret_5d_%"].toFixed(2)}%</span></div>
            <div className="d">5-day forecast · {Math.abs(a.forecasts[m].z_score).toFixed(2)}σ move · target {a.forecasts[m].path[4].toFixed(2)}</div>
          </div>
        ))}
      </div>
      <div className="card" style={{ marginTop: 14 }}>
        <Legend items={[{ key: "a", label: "Last 60 days", color: "var(--ink)" }, ...MODELS.map((m) => ({ key: m, label: NAME[m] + " path", color: COLOR[m], dash: true })), { key: "b", label: "LSTM 80% / 95% range", color: COLOR.lstm }]} />
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
            <Area dataKey="b95" name="95% range" stroke="none" fill={COLOR.lstm} fillOpacity={0.14} isAnimationActive={false} />
            <Area dataKey="b80" name="80% range" stroke="none" fill={COLOR.lstm} fillOpacity={0.3} isAnimationActive={false} />
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} minTickGap={40} />
            <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} width={64} />
            <ReferenceLine x={a.last_date} stroke="var(--axis)" strokeDasharray="3 3" label={{ value: "today", position: "insideTopRight", fill: "var(--muted)", fontSize: 12 }} />
            <Tooltip content={<TT fmt={(v: any) => Array.isArray(v) ? `${v[0].toFixed(2)} – ${v[1].toFixed(2)}` : v.toFixed(2)} />} />
            <Line dataKey="actual" name="Close" stroke="var(--ink)" strokeWidth={2} dot={false} isAnimationActive={false} />
            {MODELS.map((m) => <Line key={m} dataKey={m} name={NAME[m]} stroke={COLOR[m]} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} connectNulls isAnimationActive={false} />)}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {L.assets.some((x: any) => x.ranking) && <Ranking L={L} onPick={setT} />}
      <h2>All assets — 5-day forecast (%)</h2>
      <div className="card scroll">
        <table>
          <thead><tr><th>Asset</th>{MODELS.map((m) => <th key={m}>{NAME[m]}</th>)}<th>Consensus</th></tr></thead>
          <tbody>
            {table.map((x: any) => (
              <tr key={x.t} style={{ cursor: "pointer" }} onClick={() => setT(x.t)}>
                <td>{x.t}</td>
                {MODELS.map((m) => <td key={m} className={x[m] > 0 ? "tag-up" : "tag-down"} style={{ fontWeight: 500 }}>{x[m] == null ? "—" : (x[m] > 0 ? "+" : "") + x[m].toFixed(2)}</td>)}
                <td>{consensus(x)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Ranking({ L, onPick }: { L: any; onPick: (t: string) => void }) {
  const regions: [string, string][] = [["IN", "India (NSE)"], ["US", "US stocks & commodities"]];
  return (
    <>
      <h2>Today's market-neutral ranking (the trading signal)</h2>
      <p className="lead">The market-neutral LSTM (our best model, test IC 0.017, long/short Sharpe 1.3 on the test period) scores each asset's expected 5-day performance <i>relative to its peers</i>. A desk would go long the top 20% and short the bottom 20% in each market, so the book is hedged against the market's direction.</p>
      <div className="grid g2">
        {regions.map(([r, label]) => {
          const xs = L.assets.filter((a: any) => a.ranking?.region === r).sort((a: any, b: any) => b.ranking.score - a.ranking.score);
          if (!xs.length) return null;
          const max = Math.max(...xs.map((a: any) => Math.abs(a.ranking.score)), 1e-9);
          return (
            <div className="card" key={r}>
              <h3 style={{ marginTop: 0 }}>{label}</h3>
              {xs.map((a: any) => {
                const v = a.ranking.score, act = a.ranking.action;
                return (
                  <div key={a.ticker} onClick={() => onPick(a.ticker)} style={{ display: "grid", gridTemplateColumns: "96px 1fr 56px", alignItems: "center", gap: 8, cursor: "pointer", padding: "2px 0" }}>
                    <span style={{ fontSize: 13 }}>{a.ticker}</span>
                    <div style={{ position: "relative", height: 10 }}>
                      <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--axis)" }} />
                      <div style={{ position: "absolute", top: 1, height: 8, borderRadius: 4, background: act === "LONG" ? "var(--s3)" : act === "SHORT" ? "var(--s2)" : "var(--base)",
                        left: v >= 0 ? "50%" : `${50 - (Math.abs(v) / max) * 50}%`, width: `${(Math.abs(v) / max) * 50}%` }} />
                    </div>
                    <span style={{ fontSize: 12, fontWeight: 600, textAlign: "right" }} className={act === "LONG" ? "tag-up" : act === "SHORT" ? "tag-down" : ""}>{act}</span>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ Training */
function Training({ S }: { S: any }) {
  const [m, setM] = useState<ModelKey>("lstm");
  const runs = S.search.filter((r: any) => r.model === m);
  const maxEp = Math.max(1, ...runs.map((r: any) => r.history.length));
  const rows = Array.from({ length: maxEp }, (_, e) => {
    const r: Dict = { epoch: e + 1 };
    runs.forEach((x: any, i: number) => { if (x.history[e]) r["c" + i] = x.history[e].val_loss; });
    return r;
  });
  const best = runs.reduce((a: any, b: any) => (a && a.val_loss < b.val_loss ? a : b), null);
  return (
    <>
      <h1>Hyper-parameter search</h1>
      <p className="lead">6 configurations per architecture trained on 2007–2021 and scored on 2022–2023 validation loss (MSE on volatility-normalised 5-day returns). Early stopping with patience 4; the best configuration is retrained with 3 seeds.</p>
      <div className="controls"><Seg value={m} onChange={(v) => setM(v as ModelKey)} options={MODELS.map((x) => [x, NAME[x]])} /></div>
      <div className="grid g2">
        <div className="card">
          <h3>Validation loss per epoch</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rows} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="epoch" tickLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(3)} width={56} />
              <Tooltip content={<TT fmt={(v: number) => v.toFixed(4)} />} />
              {runs.map((x: any, i: number) => <Line key={i} dataKey={"c" + i} name={`config ${i}`} stroke={COLOR[m]} strokeOpacity={x === best ? 1 : 0.3} strokeWidth={x === best ? 2.5 : 1.5} dot={false} isAnimationActive={false} />)}
            </LineChart>
          </ResponsiveContainer>
          <div className="sub">Bold line = selected configuration.</div>
        </div>
        <div className="card scroll">
          <h3>Configurations</h3>
          <table>
            <thead><tr><th>Config</th><th>Params</th><th>Epochs</th><th>Val loss</th><th>Val DA</th><th>Val IC</th></tr></thead>
            <tbody>
              {runs.map((x: any, i: number) => (
                <tr key={i} className={x === best ? "best" : ""}>
                  <td style={{ whiteSpace: "normal", fontSize: 12 }}>{Object.entries(x.cfg).filter(([k]) => !["n_heads", "d_ff", "attn", "factor"].includes(k)).map(([k, v]) => `${k}=${v}`).join(" ")}</td>
                  <td>{(x.params / 1000).toFixed(0)}k</td><td>{x.epochs}</td><td>{x.val_loss.toFixed(4)}</td><td>{f2(x.val["DA_%"])}%</td><td>{f2(x.val.IC, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
