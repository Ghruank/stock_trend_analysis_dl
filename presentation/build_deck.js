// Builds presentation/Stock_Transformers.pptx from website/public/data/summary.json + report/figures.
// node presentation/build_deck.js   (needs pptxgenjs; NODE_PATH may point at a folder that has it)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "website/public/data/summary.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "report/figures", f);
const SHOT = (f) => path.join(__dirname, "shots", f);

// ---------------------------------------------------------------- theme (plain Office look)
const THEME = {
  name: "Stock Transformers",
  headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: {
    dk1: "1F1F1F", lt1: "FFFFFF", dk2: "44546A", lt2: "F2F2F2",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "898781", accent5: "7A4FD0", accent6: "C9A227",
    hlink: "2A78D6", folHlink: "7A4FD0",
  },
};
const HEX = { lstm: "2A78D6", transformer: "EB6834", informer: "1BAF7A", ridge: "898781", combo: "7A4FD0", naive: "C3C2B7" };
const NAME = { lstm: "LSTM", transformer: "Transformer", informer: "Informer", ridge: "Ridge", combo: "Ensemble", naive: "Naive" };
const M = ["lstm", "transformer", "informer"];
const INK = "1F1F1F", INK2 = "595959", MUTED = "8C8C8C", GRID = "E3E3E3", TINT = "F4F6F9";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Stock Transformers";
pres.author = "Khush Agrawal, Yashvi Mehta, Ghruank Kothare";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- helpers on data
const summ = (m, fs) => S.summary.find((r) => r.model === m && r.features === fs);
const get = (m, fs, k, ens = true) => {
  const r = summ(m, fs);
  if (!r) return null;
  const src = ens && r.ensemble ? r.ensemble : r.mean;
  return src[k] == null ? null : src[k];
};
const fx = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));
const B = S.baselines;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 11.9, h: 1.2, fontSize: 48, bold: true, color: INK, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 2.85, w: 11.9, h: 0.9, fontSize: 22, color: INK2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: INK, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.35, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

const T = (slide, text, o) => slide.addText(text, { isTextBox: true, fontSize: 14, color: INK, margin: 0, ...o });
const card = (slide, x, y, w, h, name, fill = TINT) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
const badge = (slide, x, y, label, color, d = 0.42) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: "badge " + label });
  T(slide, label, { x, y, w: d, h: d, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
};
const arrowR = (slide, x, y, w = 0.35) =>
  slide.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h: 0.28, fill: { color: "BFBFBF" }, line: { color: "BFBFBF" }, objectName: "arrow" });
const statTile = (slide, x, y, w, value, label, color, name) => {
  card(slide, x, y, w, 1.35, name);
  T(slide, value, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.72, fontSize: 36, bold: true, color, valign: "middle" });
  T(slide, label, { x: x + 0.2, y: y + 0.84, w: w - 0.4, h: 0.4, fontSize: 12, color: INK2, valign: "top" });
};
const chartBase = () => ({
  catAxisLabelColor: INK2, valAxisLabelColor: INK2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: INK,
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: true, valAxisLineShow: false,
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: INK, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: INK2,
});

// ================================================================= 1. title
{
  pres.addSection({ title: "Introduction" });
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  s.addText("Stock Transformers", { placeholder: "title" });
  s.addText("LSTM vs Transformer vs Informer for stock-market prediction, with engineered features and calibrated forecast ranges", { placeholder: "body" });
  T(s, "Khush Agrawal  ·  Yashvi Mehta  ·  Ghruank Kothare      Deep Learning · Final Project · November 2026",
    { x: 0.7, y: 3.85, w: 11.9, h: 0.4, fontSize: 14, color: MUTED });
  const tiles = [
    ["49", "assets: Indian large & small caps, US mega caps, commodities", HEX.lstm],
    ["87", "engineered features in 10 families, from 8 basic inputs", HEX.transformer],
    ["3", "architectures, same data, same protocol, 3 seeds each", HEX.informer],
    ["21 yrs", "of daily data, 2005 → Nov 2026; unseen test Jan 24 – Nov 26", HEX.combo],
  ];
  tiles.forEach(([v, l, c], i) => statTile(s, 0.7 + i * 3.02, 4.85, 2.8, v, l, c, "title tile " + i));
  s.addNotes("Topic: which deep sequence architecture (LSTM, Transformer or Informer) works best for stock-market prediction, under an honest evaluation, and what actually helps.");
}

// ================================================================= 2. abstract / problem
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Abstract: honest forecasts, with a range", { placeholder: "title" });
  const cov = S.cone.coverage.lstm;
  const rows = [
    ["1", "Problem", "Price charts make any model look perfect (tomorrow ≈ today). A useful forecast must beat the random walk and say how sure it is.", "C0504D"],
    ["2", "Approach", "49 assets, 87 stationary features, LSTM vs Transformer vs Informer, chronological unseen test Jan 2024 – Nov 2026, 3 seeds each.", HEX.lstm],
    ["3", "Finding", `Market direction is a coin flip, but the LSTM learns which asset beats its peers, and its 80% range holds the real price ${fx(cov["80"].all, 1)}% of the time.`, HEX.informer],
  ];
  rows.forEach(([n, h, b, c], i) => {
    const y = 1.45 + i * 1.75;
    badge(s, 0.6, y + 0.05, n, c);
    T(s, h, { x: 1.2, y, w: 5.2, h: 0.45, fontSize: 20, bold: true });
    T(s, b, { x: 1.2, y: y + 0.48, w: 5.3, h: 1.15, fontSize: 14, color: INK2, valign: "top" });
  });
  s.addImage({ path: path.join(__dirname, "..", "report", "figures", "fig_cone_deck.png"), x: 6.85, y: 1.3, w: 6.0, h: 4.05,
    altText: "TCS weekly 5-day forecast cones from the LSTM with the actual price drawn over them" });
  statTile(s, 6.85, 5.45, 2.9, fx(cov["80"].all, 1) + "%", "of real prices inside the 80% range (target 80)", HEX.lstm, "tile cov80");
  statTile(s, 9.95, 5.45, 2.9, fx(cov["95"].all, 1) + "%", "inside the 95% range (target 95), test 2024–26", HEX.combo, "tile cov95");
  s.addNotes("Each forecast is a 5-day path plus a range. The range width is calibrated only on 2022-23 validation data, scaled by the asset's current volatility. On the unseen test period, the real price lands inside the 80% band 79.8% of the time and inside the 95% band 94.8%, so the ranges are honest.");
}

// ================================================================= 3. implementation
{
  pres.addSection({ title: "Method" });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Implementation: one pipeline, three architectures", { placeholder: "title" });
  const steps = [
    ["Frozen data", "yfinance snapshot\n49 assets + 7 market series"],
    ["Cleaning", "bad ticks, demergers,\nfutures rolls"],
    ["87 features", "10 families,\nall stationary"],
    ["Train-only scaling", "per asset z-score,\nclip ±5σ"],
    ["96-day windows", "target: next 5\nreturns / σ"],
    ["Colab T4 GPU", "search → 3 seeds\n→ ensemble"],
    ["Evaluate 2024–26", "DA · IC · R²oos\nMAPE · Sharpe"],
  ];
  const w = 1.52, gap = 0.2;
  steps.forEach(([h, b], i) => {
    const x = 0.6 + i * (w + gap);
    card(s, x, 1.35, w, 1.45, "step " + h, i === 2 ? "E8F0FB" : TINT);
    T(s, h, { x: x + 0.1, y: 1.43, w: w - 0.2, h: 0.45, fontSize: 13, bold: true, align: "center", valign: "middle" });
    T(s, b, { x: x + 0.08, y: 1.9, w: w - 0.16, h: 0.8, fontSize: 11, color: INK2, align: "center", valign: "top" });
    if (i < steps.length - 1) arrowR(s, x + w + 0.005, 1.94, 0.19);
  });
  // architecture cards
  const archs = [
    ["lstm", "LSTM", ["Input proj. + asset embedding", "Stacked LSTM (64–128 units)", "Last state ‖ temporal attention", "MLP → 5 returns"], "≈ 4 s / epoch"],
    ["transformer", "Transformer", ["Input proj. + Time2Vec + asset emb.", "Pre-LN encoder, 4-head self-attention", "Last token ‖ mean pool", "MLP → 5 returns"], "≈ 17 s / epoch"],
    ["informer", "Informer", ["Conv token + temporal emb. (mo/day/wd)", "ProbSparse attention + distilling 96→48→24", "Decoder: 48-step start token + 5 slots", "Generative output → 5 returns"], "≈ 33 s / epoch"],
  ];
  archs.forEach(([k, h, blocks, cost], i) => {
    const x = 0.6 + i * 4.1, y = 3.15, cw = 3.85;
    card(s, x, y, cw, 3.85, "arch " + h);
    badge(s, x + 0.2, y + 0.18, h[0], HEX[k]);
    T(s, h, { x: x + 0.75, y: y + 0.16, w: 2.0, h: 0.45, fontSize: 18, bold: true, valign: "middle" });
    T(s, cost, { x: x + 2.4, y: y + 0.16, w: 1.25, h: 0.45, fontSize: 11, color: INK2, align: "right", valign: "middle" });
    blocks.forEach((b, j) => {
      const by = y + 0.82 + j * 0.74;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: by, w: cw - 0.5, h: 0.5, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: HEX[k], width: 1.25 }, objectName: h + " block " + j });
      T(s, b, { x: x + 0.3, y: by, w: cw - 0.6, h: 0.5, fontSize: 12, align: "center", valign: "middle" });
      if (j < blocks.length - 1)
        s.addShape(pres.shapes.DOWN_ARROW, { x: x + cw / 2 - 0.09, y: by + 0.52, w: 0.18, h: 0.2, fill: { color: "BFBFBF" }, line: { color: "BFBFBF" }, objectName: "down arrow" });
    });
  });
  s.addNotes("Everything is built locally (download, cleaning, features, scaling, dataset.npz) and uploaded once to Colab. All three models share the data sampler, asset embedding, AdamW + warm-up/cosine, mixed precision and early stopping, so differences come from the architecture alone.");
}

// ================================================================= 4. innovation
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Innovation: market-neutral targets + 87 engineered features", { placeholder: "title" });
  // left: target engineering diagram
  card(s, 0.6, 1.35, 5.9, 5.55, "target card");
  T(s, "Ask an easier, tradeable question", { x: 0.85, y: 1.5, w: 5.4, h: 0.45, fontSize: 18, bold: true });
  const flow = [
    ["Raw target", "next 5 daily returns ÷ volatility", "BFBFBF"],
    ["Subtract the peer mean", "same region (India / US+commod.), same day", HEX.combo],
    ["Model learns", "which asset will BEAT its peers", HEX.lstm],
    ["Checkpoint on val. correlation", "not MSE: scale no longer matters", HEX.transformer],
    ["Daily ranking", "long top 20%, short bottom 20% per market", HEX.informer],
  ];
  flow.forEach(([h, b, c], i) => {
    const y = 2.1 + i * 0.93;
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: y + 0.12, w: 0.36, h: 0.36, fill: { color: c }, line: { color: c }, objectName: "flow dot " + i });
    T(s, h, { x: 1.45, y, w: 4.9, h: 0.36, fontSize: 15, bold: true, valign: "middle" });
    T(s, b, { x: 1.45, y: y + 0.36, w: 4.9, h: 0.34, fontSize: 12, color: INK2, valign: "top" });
    if (i < flow.length - 1)
      s.addShape(pres.shapes.LINE, { x: 1.08, y: y + 0.5, w: 0, h: 0.55, line: { color: "BFBFBF", width: 1.5 }, objectName: "flow line " + i });
  });
  // right: ridge feature ablation
  const R = S.ridge_ablation;
  if (R) {
    const g = R.groups.map((x) => ({ l: x.n > 60 ? "only 8 basic features" : "drop " + x.group.replace("_", " "), d: (x.IC - R.ref.IC) * 1000 }))
      .sort((a, b) => b.d - a.d);
    s.addChart(pres.charts.BAR, [{ name: "Δ test IC × 1000", labels: g.map((x) => x.l), values: g.map((x) => +x.d.toFixed(2)) }], {
      x: 6.8, y: 1.3, w: 6.0, h: 5.0, barDir: "bar", chartColors: ["BFBFBF"], invertedColors: [HEX.lstm], barGapWidthPct: 45, catAxisLabelPos: "low",
      showTitle: true, title: `Which feature families carry the signal? (Δ IC × 1000 when removed; full set IC = ${R.ref.IC.toFixed(3)})`,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: false, ...chartBase(), catAxisLabelFontSize: 11,
      objectName: "feature ablation chart",
    });
    T(s, "Blue: removing the family hurts. Market context, volatility, cross-sectional ranks and oscillators (engineered families) carry the signal.",
      { x: 6.9, y: 6.35, w: 5.8, h: 0.55, fontSize: 12, color: INK2 });
  }
  s.addNotes("Most of an asset's daily move is the market's move, which nobody can predict. Removing it leaves the relative part, which our features can explain. The ablation uses a linear probe so that each family's contribution is measured cleanly.");
}

// ================================================================= 5. results
{
  pres.addSection({ title: "Results" });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Results: direction is a coin flip, ranking is learnable", { placeholder: "title" });
  const ms = M.concat(summ("combo", "xs_full") ? ["combo"] : []);
  const labels = ms.map((m) => NAME[m]).concat(["Ridge"]);
  const raw = ms.map((m) => (m === "combo" ? 0 : get(m, "full", "IC") ?? 0)).concat([B.ridge?.IC ?? 0]);
  const xs = ms.map((m) => get(m, "xs_full", "IC") ?? 0).concat([B.ridge_xs?.IC ?? 0]);
  s.addChart(pres.charts.BAR, [
    { name: "Raw return target", labels, values: raw.map((v) => +v.toFixed(4)) },
    { name: "Market-neutral target", labels, values: xs.map((v) => +v.toFixed(4)) },
  ], {
    x: 0.6, y: 1.3, w: 7.4, h: 5.6, barDir: "col", barGapWidthPct: 55, chartColors: ["BFBFBF", HEX.combo],
    showTitle: true, title: "Information coefficient on 2024–26 test (daily rank correlation, higher = better)",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: true, legendPos: "b",
    valAxisLabelFormatCode: "0.00", ...chartBase(), objectName: "IC chart",
  });
  const daMax = Math.max(...M.map((m) => get(m, "full", "DA_%") ?? 0));
  const bestXs = ms.map((m) => [m, get(m, "xs_full", "IC")]).filter((x) => x[1] != null).sort((a, b) => b[1] - a[1])[0] || ["lstm", null];
  const lsC = get(bestXs[0], "xs_full", "Sharpe_LS");
  statTile(s, 8.35, 1.35, 4.4, fx(daMax, 1) + "%", "best next-day direction accuracy of any net (coin flip = 50%)", "C0504D", "tile DA");
  statTile(s, 8.35, 2.9, 4.4, fx(bestXs[1], 3), `best IC with market-neutral target (${NAME[bestXs[0]]}); was ≈ 0.00 before`, HEX.combo, "tile IC");
  statTile(s, 8.35, 4.45, 4.4, fx(lsC, 2), `gross Sharpe of the ${NAME[bestXs[0]]} long/short book (top vs bottom 20%), 2024–26`, HEX.informer, "tile LS");
  T(s, `Price error ties the random walk (MAPE ${fx(get("lstm", "full", "MAPE_%"), 2)}% vs ${fx(B.naive["MAPE_%"], 2)}%). Ridge on the same features is still the best ranker: most of the signal is linear.`,
    { x: 8.35, y: 6.0, w: 4.4, h: 0.85, fontSize: 12, color: INK2 });
  s.addNotes("Left: with the raw target every net's IC is about zero. With the market-neutral target all three become positive and the ensemble is best. Right: direction accuracy stays near 50%, which is the honest answer for daily market direction.");
}

// ================================================================= 6. live demo
{
  pres.addSection({ title: "Demo" });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo" });
  s.addText("Live demo: tomorrow's forecasts, from today's prices", { placeholder: "title" });
  const shot = SHOT("live.png");
  if (fs.existsSync(shot)) {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.3, w: 7.6, h: 4.75, fill: { color: "FFFFFF" }, line: { color: "D9D9D9", width: 1 }, objectName: "shot frame" });
    s.addImage({ path: shot, x: 0.62, y: 1.32, w: 7.56, h: 4.71, sizing: { type: "cover", w: 7.56, h: 4.71 }, altText: "Website live demo tab" });
  }
  T(s, "Interactive website (React + Recharts): results, feature engineering, forecast cones and prediction explorer for all 49 assets, backtest, live forecasts",
    { x: 0.6, y: 6.15, w: 7.6, h: 0.75, fontSize: 12, color: INK2 });
  // live table
  const L = S.live;
  if (L) {
    const pick = ["RELIANCE", "TCS", "HDFCBANK", "YESBANK", "NVDA", "TSLA", "GOLD", "CRUDE"];
    const hasRank = L.assets.some((x) => x.ranking);
    const rows = [[{ text: "Asset", options: { bold: true } }]
      .concat(M.map((m) => ({ text: { lstm: "LSTM", transformer: "Transf.", informer: "Inform." }[m], options: { bold: true, align: "right", color: HEX[m] } })))
      .concat(hasRank ? [{ text: "Rank", options: { bold: true, align: "center" } }] : [])];
    pick.forEach((t) => {
      const a = L.assets.find((x) => x.ticker === t);
      if (!a) return;
      rows.push([{ text: t }]
        .concat(M.map((m) => {
          const v = a.forecasts[m]?.["ret_5d_%"];
          return { text: v == null ? "–" : (v > 0 ? "+" : "") + v.toFixed(1) + "%", options: { align: "right", color: v > 0 ? "1B7F4B" : "B03A2E" } };
        }))
        .concat(hasRank ? [{ text: a.ranking?.action ?? "–", options: { align: "center", bold: true, color: a.ranking?.action === "LONG" ? "1B7F4B" : a.ranking?.action === "SHORT" ? "B03A2E" : MUTED } }] : []));
    });
    T(s, `Predicted 5-day return after ${L.generated_for}`, { x: 8.55, y: 1.3, w: 4.2, h: 0.4, fontSize: 14, bold: true });
    s.addTable(rows, { x: 8.55, y: 1.8, w: 4.2, colW: hasRank ? [1.15, 0.72, 0.72, 0.72, 0.89] : [1.3, 0.97, 0.97, 0.96], fontSize: 11, color: INK, fontFace: "Calibri",
      border: { type: "solid", pt: 0.5, color: "E3E3E3" }, rowH: 0.36, fill: { color: "FFFFFF" }, objectName: "live table" });
    T(s, "Columns: each model's predicted 5-day return. Rank: market-neutral LSTM → LONG top 20% / SHORT bottom 20% within its market.\nRun: python live_predict.py --refresh",
      { x: 8.55, y: 5.3, w: 4.2, h: 1.1, fontSize: 11, color: INK2 });
  }
  s.addNotes("Demo: open the website's Live demo tab, pick an asset, show the three models' 5-day paths. Then show the Prediction explorer and the Backtest tab.");
}

// ================================================================= 7. real-life use + verdict
{
  pres.addSection({ title: "Conclusion" });
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
  s.addText("Real-life use, and which architecture wins", { placeholder: "title" });
  T(s, "A quant desk's evening routine", { x: 0.6, y: 1.3, w: 12, h: 0.45, fontSize: 18, bold: true });
  const steps = [["3:30 pm", "NSE closes; prices refreshed"], ["Features", "87 signals rebuilt per asset"], ["LSTM", "scores every asset vs its peers"],
    ["Rank", "within India and US"], ["Trade", "long top 20%, short bottom 20%: market-neutral"], ["Monitor", "flag big predicted 5-day moves for risk"]];
  const w = 1.88;
  steps.forEach(([h, b], i) => {
    const x = 0.6 + i * (w + 0.18);
    card(s, x, 1.85, w, 1.35, "use " + h, i === 4 ? "EFE9FA" : TINT);
    T(s, h, { x: x + 0.1, y: 1.93, w: w - 0.2, h: 0.4, fontSize: 15, bold: true, align: "center" });
    T(s, b, { x: x + 0.1, y: 2.35, w: w - 0.2, h: 0.75, fontSize: 12, color: INK2, align: "center", valign: "top" });
    if (i < steps.length - 1) arrowR(s, x + w + 0.005, 2.38, 0.17);
  });
  // verdict cards
  const v = [
    ["lstm", `LSTM: the winner (IC ${fx(get("lstm", "xs_full", "IC"), 3)})`, "Best IC, peer accuracy and Sharpe; most stable across seeds; ≈ 8× cheaper than the Informer."],
    ["transformer", `Transformer: weaker (IC ${fx(get("transformer", "xs_full", "IC"), 3)})`, "Learns a signal, but 5× more seed-dependent; self-attention adds variance on 96 daily steps."],
    ["informer", `Informer: no ranking skill (IC ${fx(get("informer", "xs_full", "IC"), 3)})`, "ProbSparse and distilling are built for 1000+ steps, not 96; slowest to train."],
    ["combo", `Averaging all 3: worse than LSTM (IC ${fx(get("combo", "xs_full", "IC"), 3)})`, "Ensembles only help when every member has skill; the Informer adds noise."],
  ];
  v.forEach(([k, h, b], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 3.55 + Math.floor(i / 2) * 1.5;
    card(s, x, y, 5.95, 1.3, "verdict " + k);
    badge(s, x + 0.2, y + 0.2, String(i + 1), HEX[k]);
    T(s, h, { x: x + 0.8, y: y + 0.15, w: 5.0, h: 0.45, fontSize: 16, bold: true, valign: "middle" });
    T(s, b, { x: x + 0.8, y: y + 0.6, w: 5.0, h: 0.6, fontSize: 12.5, color: INK2, valign: "top" });
  });
  T(s, "Takeaway: use an LSTM, but the target, the features and an honest evaluation matter more than the architecture.",
    { x: 0.6, y: 6.6, w: 12.1, h: 0.4, fontSize: 15, italic: true, color: HEX.combo });
  s.addNotes("Use case: a market-neutral long/short book or a risk early-warning list, refreshed every evening. Caveat: daily rebalancing costs eat much of the gross edge, so real deployment needs slower rebalancing.");
}

(async () => {
  const out = path.join(__dirname, "Stock_Transformers.pptx");
  await pres.writeFile({ fileName: out });
  const skill = process.env.PPTX_SKILL;
  if (skill) {
    const { applyTheme } = require(path.join(skill, "scripts/apply_theme.js"));
    await applyTheme(out, THEME);
  }
  console.log("wrote", out);
})();
