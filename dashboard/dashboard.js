const history = [];
const maxHistory = 72;
const priceElement = document.querySelector("#mark-price");
const priceUpdatedElement = document.querySelector("#price-updated");
const chart = document.querySelector("#price-chart");
const chartEmpty = document.querySelector("#chart-empty");
const chartContext = chart.getContext("2d");
let lastMarkPrice = null;
let signalSocket;
let signalRetry;
let pingTimer;
let selectedAsset = "XAUUSDT";

const priceFormat = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function setFeedStatus(feed, status, label) {
  document.querySelector(`#${feed}-status-dot`).className = `status-dot status-dot--${status}`;
  document.querySelector(`#${feed}-status-text`).textContent = label;
}

function updateSignal(signal) {
  const panel = document.querySelector("#signal-panel");
  const label = signal.label || "FLAT";
  const type = label.startsWith("LONG") ? "long" : label.startsWith("SHORT") ? "short" : "flat";
  const symbol = type === "long" ? "↑" : type === "short" ? "↓" : "—";
  panel.className = `signal-panel signal-panel--${type}`;
  document.querySelector("#signal-symbol").textContent = symbol;
  document.querySelector("#signal-label").textContent = label;
  document.querySelector("#signal-caption").textContent = `Calculated from the ${new Date(signal.time).toUTCString()} close`;
  document.querySelector("#signal-score").textContent = Number(signal.score).toFixed(3);
  document.querySelector("#candle-close").textContent = priceFormat.format(signal.close);
  document.querySelector("#signal-time").textContent = new Date(signal.time).toLocaleString([], {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  });
  const boundedScore = Math.max(-2, Math.min(2, Number(signal.score)));
  document.querySelector("#score-marker").style.left = `${((boundedScore + 2) / 4) * 100}%`;
}

function setHistory(candles) {
  history.splice(0, history.length, ...candles.slice(-maxHistory));
  drawChart();
  updateLastCandle();
}

function addCandle(candle) {
  const time = new Date(candle.time).getTime();
  const last = history[history.length - 1];
  if (last && new Date(last.time).getTime() === time) history[history.length - 1] = candle;
  else history.push(candle);
  if (history.length > maxHistory) history.splice(0, history.length - maxHistory);
  drawChart();
  updateLastCandle();
}

function updateLastCandle() {
  const last = history[history.length - 1];
  if (!last) return;
  document.querySelector("#last-candle-time").textContent = `Last close ${new Date(last.time).toLocaleString([], {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
  })}`;
  if (lastMarkPrice === null) {
    priceElement.textContent = priceFormat.format(last.close);
    document.querySelector("#price-kind").textContent = "LAST 1H CLOSE";
    priceUpdatedElement.textContent = `Close ${new Date(last.time).toLocaleString([], {
      month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short",
    })}`;
  }
  chartEmpty.hidden = true;
}

function drawChart() {
  const bounds = chart.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.round(bounds.width * ratio);
  chart.height = Math.round(bounds.height * ratio);
  chartContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  chartContext.clearRect(0, 0, bounds.width, bounds.height);
  if (history.length < 2) return;

  const values = history.map((item) => Number(item.close));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = { top: 13, right: 8, bottom: 24, left: 68 };
  const width = bounds.width - padding.left - padding.right;
  const height = bounds.height - padding.top - padding.bottom;
  const spread = high - low || Math.max(high * 0.001, 1);
  const min = low - spread * 0.12;
  const max = high + spread * 0.12;
  const x = (index) => padding.left + (index / (values.length - 1)) * width;
  const y = (value) => padding.top + ((max - value) / (max - min)) * height;
  const styles = getComputedStyle(document.documentElement);
  const gridColor = styles.getPropertyValue("--line").trim();
  const lineColor = styles.getPropertyValue("--green").trim();
  const labelColor = styles.getPropertyValue("--muted").trim();

  chartContext.font = '10px "DM Mono", monospace';
  chartContext.textBaseline = "middle";
  for (let row = 0; row <= 4; row += 1) {
    const yPos = padding.top + (height * row) / 4;
    const value = max - ((max - min) * row) / 4;
    chartContext.beginPath();
    chartContext.moveTo(padding.left, yPos);
    chartContext.lineTo(bounds.width - padding.right, yPos);
    chartContext.strokeStyle = gridColor;
    chartContext.lineWidth = 1;
    chartContext.stroke();
    chartContext.fillStyle = labelColor;
    chartContext.textAlign = "left";
    chartContext.fillText(priceFormat.format(value), 0, yPos);
  }

  chartContext.beginPath();
  values.forEach((value, index) => {
    if (index === 0) chartContext.moveTo(x(index), y(value));
    else chartContext.lineTo(x(index), y(value));
  });
  chartContext.strokeStyle = lineColor;
  chartContext.lineWidth = 2;
  chartContext.lineJoin = "round";
  chartContext.lineCap = "round";
  chartContext.stroke();

  const first = history[0];
  const last = history[history.length - 1];
  chartContext.fillStyle = labelColor;
  chartContext.textAlign = "left";
  chartContext.textBaseline = "bottom";
  chartContext.fillText(new Date(first.time).toLocaleDateString([], { month: "short", day: "2-digit", timeZone: "UTC" }), padding.left, bounds.height - 2);
  chartContext.textAlign = "right";
  chartContext.fillText(new Date(last.time).toLocaleDateString([], { month: "short", day: "2-digit", timeZone: "UTC" }), bounds.width - padding.right, bounds.height - 2);
}

function setResearchStatus(status, label) {
  document.querySelector("#research-status-dot").className = `status-dot status-dot--${status}`;
  document.querySelector("#research-status-text").textContent = label;
}

function formatPercent(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
}

function formatMetric(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function formatMoney(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number < 0 ? "−" : ""}${priceFormat.format(Math.abs(number))}` : "—";
}

function renderPaperAccount(account) {
  document.querySelector("#paper-equity").textContent = formatMoney(account.equity);
  document.querySelector("#paper-cash").textContent = formatMoney(account.cash);
  document.querySelector("#paper-return").textContent = formatPercent(account.net_return);
  document.querySelector("#paper-realized").textContent = formatMoney(account.realized_pnl);
  document.querySelector("#paper-fees").textContent = formatMoney(account.fees_paid);
  document.querySelector("#paper-drawdown").textContent = formatPercent(account.max_drawdown);
  document.querySelector("#paper-exposure").textContent = formatMoney(account.gross_exposure);
  document.querySelector("#paper-position-count").textContent = String(account.positions.length);
  document.querySelector("#paper-status").textContent = account.killed ? "HALTED" : "PAPER";
  document.querySelector("#paper-state").textContent = account.killed ? "Kill switch active" : "Ready for orders";
  document.querySelector("#paper-state").className = account.killed ? "paper-state--alert" : "";

  const positions = document.querySelector("#paper-positions");
  positions.replaceChildren();
  if (!account.positions.length) {
    positions.innerHTML = '<tr><td colspan="7" class="paper-empty">No open positions</td></tr>';
  } else account.positions.forEach((position) => {
    const row = document.createElement("tr");
    row.innerHTML = `<th>${position.symbol}</th><td>${formatMetric(position.quantity, 6)}</td><td>${formatMoney(position.average_price)}</td><td>${formatMoney(position.last_price)}</td><td>${formatMoney(position.notional)}</td><td>${formatMoney(position.unrealized_pnl)}</td><td><button class="table-action" type="button" data-close-symbol="${position.symbol}">Close</button></td>`;
    positions.append(row);
  });

  const trades = document.querySelector("#paper-trades");
  trades.replaceChildren();
  if (!account.trades.length) {
    trades.innerHTML = '<tr><td colspan="7" class="paper-empty">No trades yet</td></tr>';
  } else [...account.trades].reverse().forEach((trade) => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${new Date(trade.time).toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", timeZone: "UTC" })}</td><th>${trade.symbol}</th><td>${trade.side.toUpperCase()}</td><td>${formatMetric(trade.quantity, 6)}</td><td>${formatMoney(trade.price)}</td><td>${formatMoney(trade.fee)}</td><td>${formatMoney(trade.realized_pnl)}</td>`;
    trades.append(row);
  });
}

async function refreshPaperAccount() {
  try {
    const response = await fetch("/api/paper/account");
    if (!response.ok) throw new Error(`Paper account unavailable (${response.status})`);
    renderPaperAccount(await response.json());
  } catch (error) {
    document.querySelector("#paper-notice").textContent = error.message;
  }
}

async function submitPaperOrder(event) {
  event.preventDefault();
  const notice = document.querySelector("#paper-notice");
  const payload = {
    symbol: document.querySelector("#paper-symbol").value,
    side: document.querySelector("#paper-side").value,
    quantity: Number(document.querySelector("#paper-quantity").value),
  };
  try {
    const response = await fetch("/api/paper/order", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Paper order rejected");
    renderPaperAccount(data);
    notice.textContent = `${payload.side.toUpperCase()} ${payload.quantity} ${payload.symbol} filled in paper account.`;
  } catch (error) { notice.textContent = error.message; }
}

async function closePaperPosition(symbol) {
  const response = await fetch(`/api/paper/close/${encodeURIComponent(symbol)}`, { method: "POST" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Position close rejected");
  renderPaperAccount(data);
  document.querySelector("#paper-notice").textContent = `${symbol} position closed in paper account.`;
}

function clearResearchView(symbol) {
  document.querySelector("#research-signal").textContent = "Loading history";
  document.querySelector("#research-model-name").textContent = `${symbol} · waiting for research snapshot`;
  for (const id of [
    "research-probability", "research-accuracy", "research-weight", "bt-model-return", "bt-model-sharpe",
    "bt-model-drawdown", "bt-model-hit", "bt-momentum-return", "bt-momentum-sharpe",
    "bt-momentum-drawdown", "bt-momentum-hit", "portfolio-target-vol", "portfolio-realized-vol",
    "portfolio-cap", "portfolio-current-weight", "risk-events", "research-bar-count",
  ]) document.querySelector(`#${id}`).textContent = "—";
  document.querySelector("#forward-bars").textContent = "0";
  document.querySelector("#forward-equity").textContent = "1.0000";
  document.querySelector("#forward-return").textContent = "0.00%";
  document.querySelector("#forward-drawdown").textContent = "0.00%";
  document.querySelector("#forward-hit").textContent = "—";
  document.querySelector("#risk-state").textContent = "WAITING";
  document.querySelector("#research-updated").textContent = "Waiting for new asset history";
  document.querySelector("#research-price-symbol").textContent = symbol;
  setFeedStatus("price", "waiting", "Awaiting quote");
  setFeedStatus("signal", "waiting", "Loading history");
  document.querySelector("#research-mark-price").textContent = "—";
  document.querySelector("#research-mark-time").textContent = "MARK PRICE";
  document.querySelector("#chart-symbol").textContent = symbol;
  document.querySelector("#chart-title").textContent = symbol;
  lastMarkPrice = null;
}

function renderResearchSnapshot(data) {
  if (!data || data.symbol !== selectedAsset || data.status !== "live") return;
  const report = data.report;
  const model = report.model;
  const forward = data.forward;
  const backtest = report.backtest;
  const portfolio = report.portfolio;

  document.querySelector("#research-signal").textContent = model.signal;
  document.querySelector("#research-model-name").textContent = `${model.name} · ${data.symbol} · 1h candles`;
  document.querySelector("#research-probability").textContent = formatPercent(model.probability_up);
  document.querySelector("#research-accuracy").textContent = formatPercent(model.accuracy);
  document.querySelector("#research-weight").textContent = formatMetric(model.position_weight, 3);
  document.querySelector("#backtest-window").textContent = `${report.holdout_start.slice(0, 10)} to ${report.holdout_end.slice(0, 10)}`;
  document.querySelector("#backtest-method").textContent = `${backtest.method} · ${report.train_bars.toLocaleString()} train bars / ${report.holdout_bars.toLocaleString()} holdout bars · ${backtest.commission_bps_per_side} bps commission + ${backtest.slippage_bps} bps slippage per side.`;

  const rows = [
    ["model", backtest.model],
    ["momentum", backtest.momentum],
  ];
  rows.forEach(([prefix, metrics]) => {
    document.querySelector(`#bt-${prefix}-return`).textContent = formatPercent(metrics.cum_return);
    document.querySelector(`#bt-${prefix}-sharpe`).textContent = formatMetric(metrics.sharpe);
    document.querySelector(`#bt-${prefix}-drawdown`).textContent = formatPercent(metrics.max_dd);
    document.querySelector(`#bt-${prefix}-hit`).textContent = formatPercent(metrics.hit_ratio);
  });

  document.querySelector("#portfolio-target-vol").textContent = formatPercent(portfolio.target_annual_volatility);
  document.querySelector("#portfolio-realized-vol").textContent = formatPercent(portfolio.realized_annual_volatility);
  document.querySelector("#portfolio-cap").textContent = formatMetric(portfolio.position_cap, 3);
  document.querySelector("#portfolio-current-weight").textContent = formatMetric(forward.position_weight, 3);
  document.querySelector("#risk-events").textContent = forward.risk_breaches.length ? forward.risk_breaches.join(", ") : "None";
  document.querySelector("#risk-state").textContent = forward.risk_killed ? "KILL SWITCH" : forward.risk_breaches.length ? "LIMIT ACTIVE" : "WITHIN LIMITS";
  document.querySelector("#risk-state").className = `risk-state${forward.risk_killed ? " risk-state--alert" : forward.risk_breaches.length ? " risk-state--warn" : ""}`;

  document.querySelector("#forward-bars").textContent = forward.bars.toLocaleString();
  document.querySelector("#forward-equity").textContent = formatMetric(forward.equity, 4);
  document.querySelector("#forward-return").textContent = formatPercent(forward.cumulative_return);
  document.querySelector("#forward-drawdown").textContent = formatPercent(forward.max_drawdown);
  document.querySelector("#forward-hit").textContent = forward.bars ? formatPercent(forward.hit_rate) : "Awaiting bars";
  document.querySelector("#forward-caption").textContent = forward.bars
    ? `Paper observations since this process started · latest completed bar ${forward.last_bar_time}`
    : "Forward observations start after this app session begins. Historical holdout metrics are shown separately above.";
  document.querySelector("#research-bar-count").textContent = `${report.bars.toLocaleString()} closed hourly bars · updated ${new Date(data.updated_at).toLocaleString()}`;
  document.querySelector("#research-updated").textContent = `Research refreshed ${new Date(data.updated_at).toLocaleTimeString()}`;
  document.querySelector("#research-price-symbol").textContent = data.symbol;
  setResearchStatus("online", "Research current · 1h bars");
  setFeedStatus("signal", "online", "Live · 1h close");
  setHistory(data.history || []);
  if (data.market_signal) updateSignal(data.market_signal);
}

async function loadResearchAssets() {
  const selector = document.querySelector("#asset-select");
  try {
    const response = await fetch("/api/research/assets");
    if (!response.ok) throw new Error(`Asset list unavailable (${response.status})`);
    const data = await response.json();
    selector.replaceChildren(...data.assets.map((symbol) => {
      const option = document.createElement("option");
      option.value = symbol;
      option.textContent = symbol;
      return option;
    }));
    selectedAsset = data.selected;
    selector.value = selectedAsset;
    const paperSelector = document.querySelector("#paper-symbol");
    paperSelector.replaceChildren(...data.assets.map((symbol) => {
      const option = document.createElement("option");
      option.value = symbol;
      option.textContent = symbol;
      return option;
    }));
    paperSelector.value = selectedAsset;
    document.querySelector("#research-price-symbol").textContent = selectedAsset;
    setResearchStatus("waiting", "Loading historical research");
  } catch (error) {
    setResearchStatus("offline", error.message);
  }
}

async function selectResearchAsset(symbol) {
  selectedAsset = symbol;
  clearResearchView(symbol);
  setResearchStatus("waiting", `Loading ${symbol} history`);
  try {
    const response = await fetch("/api/research/asset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Asset selection failed");
    if (data.status === "live") renderResearchSnapshot(data);
  } catch (error) {
    setResearchStatus("offline", error.message);
  }
}

function connectResearchFeed() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  signalSocket = new WebSocket(`${protocol}//${window.location.host}/ws/research`);
  signalSocket.onopen = () => {
    setResearchStatus("online", "Research feed connected");
    setFeedStatus("signal", "online", "Connected");
    clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      if (signalSocket.readyState === WebSocket.OPEN) signalSocket.send("ping");
    }, 25000);
  };
  signalSocket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "snapshot") {
      renderResearchSnapshot(message.data);
    } else if (message.type === "mark_price") {
      if (message.symbol !== selectedAsset) return;
      lastMarkPrice = Number(message.price);
      const timeLabel = `Updated ${new Date(message.time).toLocaleTimeString([], {
        hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC", timeZoneName: "short",
      })}`;
      document.querySelector("#research-mark-price").textContent = priceFormat.format(lastMarkPrice);
      document.querySelector("#research-mark-time").textContent = timeLabel;
      priceElement.textContent = priceFormat.format(lastMarkPrice);
      document.querySelector("#price-kind").textContent = "BINANCE MARK PRICE";
      priceUpdatedElement.textContent = timeLabel;
      setFeedStatus("price", "online", "Live · 5s");
    } else if (message.type === "status") {
      setResearchStatus(message.data?.status === "error" ? "offline" : "waiting", message.data?.error || `Loading ${message.data?.symbol || selectedAsset}`);
      setFeedStatus("signal", "waiting", "Updating");
    }
  };
  signalSocket.onerror = () => {
    setResearchStatus("offline", "Research feed error");
    setFeedStatus("signal", "offline", "Feed error");
  };
  signalSocket.onclose = () => {
    clearInterval(pingTimer);
    setResearchStatus("waiting", "Reconnecting");
    setFeedStatus("signal", "waiting", "Reconnecting");
    setFeedStatus("price", "waiting", "Reconnecting");
    clearTimeout(signalRetry);
    signalRetry = setTimeout(connectResearchFeed, 2500);
  };
}

function updateClock() {
  document.querySelector("#utc-clock").textContent = `UTC ${new Date().toISOString().slice(11, 19)}`;
}

const moduleGrid = document.querySelector("#module-grid");
const moduleCount = document.querySelector("#module-count");
const fileList = document.querySelector("#file-list");
const fileCount = document.querySelector("#file-count");
const fileSearch = document.querySelector("#file-search");
const sourcePath = document.querySelector("#source-path");
const sourceLanguage = document.querySelector("#source-language");
const sourceContent = document.querySelector("#source-content");
let projectFiles = [];

function activateView(viewId) {
  document.querySelectorAll(".workspace-tab").forEach((tab) => {
    const active = tab.dataset.view === viewId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".workspace-view").forEach((view) => {
    view.hidden = view.id !== viewId;
  });
  if (viewId === "market-view") requestAnimationFrame(drawChart);
}

function renderModules(modules) {
  moduleGrid.replaceChildren();
  moduleCount.textContent = `${modules.length} modules`;
  modules.forEach((module) => {
    const card = document.createElement("article");
    card.className = "module-row";

    const heading = document.createElement("div");
    heading.className = "module-row-heading";
    const title = document.createElement("h2");
    title.textContent = module.name;
    const status = document.createElement("span");
    status.className = `module-status module-status--${module.status}`;
    status.textContent = module.status.replaceAll("-", " ");
    heading.append(title, status);

    const description = document.createElement("p");
    description.textContent = module.description;
    const location = document.createElement("span");
    location.className = "module-location";
    location.textContent = module.path;
    const action = document.createElement("button");
    action.className = "source-action";
    action.type = "button";
    action.textContent = "Browse source";
    action.dataset.openPrefix = module.path;

    card.append(heading, description, location, action);
    moduleGrid.append(card);
  });
}

function renderFiles(query = "") {
  const normalizedQuery = query.trim().toLowerCase();
  const visibleFiles = projectFiles.filter((path) => path.toLowerCase().includes(normalizedQuery));
  fileCount.textContent = `${visibleFiles.length} files`;
  fileList.replaceChildren();
  visibleFiles.forEach((path) => {
    const button = document.createElement("button");
    button.className = "file-entry";
    button.type = "button";
    button.textContent = path;
    button.dataset.sourcePath = path;
    fileList.append(button);
  });
}

async function openSourceFile(path) {
  sourcePath.textContent = path;
  sourceLanguage.textContent = path.split(".").at(-1).toUpperCase();
  sourceContent.textContent = "Loading file…";
  document.querySelectorAll(".file-entry").forEach((entry) => {
    entry.classList.toggle("is-selected", entry.dataset.sourcePath === path);
  });
  try {
    const encodedPath = path.split("/").map(encodeURIComponent).join("/");
    const response = await fetch(`/api/project/files/${encodedPath}`);
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    sourceContent.textContent = await response.text();
  } catch (error) {
    sourceContent.textContent = `Could not load this file: ${error.message}`;
  }
}

async function loadProjectDirectory() {
  try {
    const [modulesResponse, filesResponse] = await Promise.all([
      fetch("/api/project/modules"),
      fetch("/api/project/files"),
    ]);
    if (!modulesResponse.ok || !filesResponse.ok) throw new Error("Project directory request failed.");
    renderModules(await modulesResponse.json());
    projectFiles = await filesResponse.json();
    renderFiles();
  } catch (error) {
    moduleCount.textContent = "Unavailable";
    moduleGrid.textContent = error.message;
    fileCount.textContent = "Unavailable";
    fileList.textContent = error.message;
  }
}

document.querySelectorAll(".workspace-tab").forEach((tab) => {
  tab.addEventListener("click", () => activateView(tab.dataset.view));
});
document.querySelector("#asset-select").addEventListener("change", (event) => selectResearchAsset(event.target.value));
document.querySelector("#paper-order-form").addEventListener("submit", submitPaperOrder);
document.querySelector("#paper-reset").addEventListener("click", async () => {
  const response = await fetch("/api/paper/reset", { method: "POST" });
  renderPaperAccount(await response.json());
  document.querySelector("#paper-notice").textContent = "Paper account reset to 10,000 USDT.";
});
document.querySelector("#paper-positions").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-close-symbol]");
  if (!button) return;
  try { await closePaperPosition(button.dataset.closeSymbol); } catch (error) { document.querySelector("#paper-notice").textContent = error.message; }
});
moduleGrid.addEventListener("click", (event) => {
  const action = event.target.closest("[data-open-prefix]");
  if (!action) return;
  const file = projectFiles.find((path) => path.startsWith(action.dataset.openPrefix));
  if (!file) return;
  activateView("files-view");
  openSourceFile(file);
});
fileList.addEventListener("click", (event) => {
  const entry = event.target.closest("[data-source-path]");
  if (entry) openSourceFile(entry.dataset.sourcePath);
});
fileSearch.addEventListener("input", () => renderFiles(fileSearch.value));
document.querySelector("#source-content").addEventListener("keydown", (event) => event.preventDefault());
loadProjectDirectory();
loadResearchAssets();
refreshPaperAccount();
setInterval(refreshPaperAccount, 5000);

new ResizeObserver(drawChart).observe(chart);
updateClock();
setInterval(updateClock, 1000);
connectResearchFeed();