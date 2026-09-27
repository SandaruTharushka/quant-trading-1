"""FastAPI app exposing health, signals, risk, and a WebSocket for live updates."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging import get_logger
from data.providers.binance import BinanceFuturesProvider
from research.live_research import ForwardPaperLedger, build_research_report
from research.signals import analyze
from paper_trading import PaperTradingAccount

log = get_logger("api")
DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
PROJECT_ROOT = DASHBOARD_DIR.parent.resolve()
PROJECT_MODULES = [
    {"name": "Market data", "path": "data/", "status": "partial", "description": "Market-data providers, historical replay, and storage."},
    {"name": "Features", "path": "features/", "status": "research", "description": "Fractional differencing, event labeling, and microstructure features."},
    {"name": "Models", "path": "models/", "status": "partial", "description": "The dashboard runs a Logistic Regression baseline; neural, limit-order-book, and reinforcement models remain research components."},
    {"name": "Research", "path": "research/", "status": "active", "description": "Live hourly baseline evaluation, signal metrics, and chronological holdout validation."},
    {"name": "Backtesting", "path": "backtesting/", "status": "active", "description": "Live dashboard compares costed out-of-sample model and momentum backtests."},
    {"name": "Portfolio", "path": "portfolio/", "status": "active", "description": "Live baseline position sizing uses volatility targeting and configured position caps."},
    {"name": "Risk", "path": "risk/", "status": "active", "description": "Forward paper ledger applies configured risk limits and reports breaches or kill-switch state."},
    {"name": "Paper trading", "path": "paper_trading.py", "status": "active", "description": "Web paper account with market orders, positions, PnL, fees, and drawdown protection."},
    {"name": "Execution", "path": "execution/", "status": "scaffold", "description": "Execution algorithms and abstract broker router; no live broker is connected."},
    {"name": "Regime detection", "path": "regime/", "status": "research", "description": "HMM and change-point regime methods."},
    {"name": "Monitoring", "path": "monitoring/", "status": "research", "description": "Anomaly and drift analysis utilities."},
    {"name": "Inference", "path": "inference/", "status": "partial", "description": "Model export and inference server components."},
    {"name": "MLOps", "path": "mlops/", "status": "partial", "description": "Model registry and walk-forward retraining utilities."},
    {"name": "API", "path": "api/", "status": "active", "description": "FastAPI dashboard, health, risk limits, and live signal feed."},
    {"name": "Tests", "path": "tests/", "status": "active", "description": "Automated tests for implemented research components."},
]
SOURCE_ROOTS = {
    "api", "backtesting", "configs", "core", "data", "deployment", "execution",
    "features", "inference", "mlops", "models", "monitoring", "portfolio",
    "regime", "research", "risk", "scripts", "tests", "dashboard",
}
SOURCE_SUFFIXES = {".css", ".html", ".js", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
ROOT_SOURCE_FILES = {"README.md", "pyproject.toml", "requirements.txt", "paper_trading.py"}
_LIVE_CLIENTS: set[WebSocket] = set()
_LIVE_STATE: dict | None = None
_RESEARCH_CLIENTS: set[WebSocket] = set()
_RESEARCH_STATE: dict = {"status": "connecting", "symbol": "XAUUSDT"}
_RESEARCH_SYMBOL = "XAUUSDT"
_RESEARCH_ASSETS: set[str] = set()
_RESEARCH_LEDGER: dict[str, ForwardPaperLedger] = {}
_RESEARCH_SNAPSHOTS: dict[str, dict] = {}
_RESEARCH_WAKE = asyncio.Event()
_MARK_PRICES: dict[str, float] = {}
_PAPER_ACCOUNT: PaperTradingAccount | None = None


def _signal_payload(close: pd.Series) -> dict:
    timestamp, score, label = analyze(close)
    return {
        "time": timestamp.isoformat(),
        "close": float(close.loc[timestamp]),
        "score": score,
        "label": label,
    }


def _candle_payload(close: pd.Series) -> list[dict]:
    return [
        {"time": timestamp.isoformat(), "close": float(price)}
        for timestamp, price in close.items()
    ]


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


async def _broadcast_live(message: dict) -> None:
    for client in list(_LIVE_CLIENTS):
        try:
            await client.send_json(message)
        except Exception:
            _LIVE_CLIENTS.discard(client)


async def _broadcast_research(message: dict) -> None:
    for client in list(_RESEARCH_CLIENTS):
        try:
            await client.send_json(message)
        except Exception:
            _RESEARCH_CLIENTS.discard(client)


def _ledger_for(symbol: str) -> ForwardPaperLedger:
    settings = get_settings().risk
    if symbol not in _RESEARCH_LEDGER:
        from risk.engine import RiskEngine, RiskLimits

        _RESEARCH_LEDGER[symbol] = ForwardPaperLedger(
            risk_engine=RiskEngine(limits=RiskLimits(
                max_drawdown=settings.max_drawdown,
                daily_loss_limit=settings.daily_loss_limit,
                var_confidence=settings.var_confidence,
                max_var=settings.max_var,
                max_cvar=settings.max_cvar,
                max_gross_leverage=settings.max_gross_leverage,
                max_position_pct=settings.max_position_pct,
                vol_circuit_breaker=settings.vol_circuit_breaker,
            ))
        )
    return _RESEARCH_LEDGER[symbol]


async def _fetch_research_assets() -> list[str]:
    global _RESEARCH_ASSETS
    if not _RESEARCH_ASSETS:
        url = f"{BinanceFuturesProvider.BASE}/fapi/v1/exchangeInfo"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        _RESEARCH_ASSETS = {
            item["symbol"] for item in response.json().get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and "PERPETUAL" in item.get("contractType", "")
        }
    return sorted(_RESEARCH_ASSETS)


async def _run_research_worker() -> None:
    global _RESEARCH_STATE
    provider = BinanceFuturesProvider()
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    last_bar_by_symbol: dict[str, str] = {}
    while True:
        _RESEARCH_WAKE.clear()
        symbol = _RESEARCH_SYMBOL
        try:
            end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            if symbol not in bars_by_symbol:
                bars = await provider.get_bars(symbol, end - timedelta(days=180), end, timeframe="1h")
            else:
                recent = await provider.get_bars(symbol, end - timedelta(hours=4), end, timeframe="1h")
                bars = pd.concat([bars_by_symbol[symbol], recent])
                bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index().tail(24 * 180)
            bars = bars.loc[bars.index < pd.Timestamp(end)]
            bars_by_symbol[symbol] = bars
            if len(bars) < 300:
                raise RuntimeError(f"Not enough closed hourly bars for {symbol} (need at least 300).")

            latest_bar = bars.index[-1].isoformat()
            if last_bar_by_symbol.get(symbol) != latest_bar:
                settings = get_settings()
                report = await asyncio.to_thread(
                    build_research_report,
                    bars,
                    target_vol=settings.risk.vol_target_annual,
                    max_position=settings.risk.max_position_pct,
                    commission_bps=settings.execution.base_commission_bps,
                    slippage_bps=settings.execution.base_slippage_bps,
                )
                ledger = _ledger_for(symbol)
                last_close = float(bars["c"].iloc[-1])
                timestamp = pd.Timestamp(bars.index[-1])
                if ledger.last_time is None:
                    ledger.last_close = last_close
                    ledger.last_time = timestamp
                    ledger.day = timestamp.date()
                    ledger.day_start_equity = ledger.equity
                    ledger.risk_engine.reset_day(ledger.equity)
                    ledger.position_weight = float(report["_latest_weight"])
                elif timestamp > ledger.last_time:
                    ledger.update(
                        symbol,
                        timestamp,
                        last_close,
                        float(report["_latest_weight"]),
                        max_position=settings.risk.max_position_pct,
                        commission_bps=settings.execution.base_commission_bps,
                        slippage_bps=settings.execution.base_slippage_bps,
                    )

                report.pop("_latest_probability", None)
                report.pop("_latest_weight", None)
                report.pop("_latest_close", None)
                report.pop("_last_bar_time", None)
                _RESEARCH_STATE = {
                    "status": "live",
                    "symbol": symbol,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "price": last_close,
                    "signal_time": latest_bar,
                    "signal": report["model"]["signal"],
                    "market_signal": _signal_payload(bars["c"].astype(float)),
                    "history": _candle_payload(bars["c"].astype(float).tail(72)),
                    "report": report,
                    "forward": ledger.snapshot(),
                }
                _RESEARCH_SNAPSHOTS[symbol] = _RESEARCH_STATE
                last_bar_by_symbol[symbol] = latest_bar
                await _broadcast_research({"type": "snapshot", "data": _RESEARCH_STATE})
            elif symbol in _RESEARCH_SNAPSHOTS:
                _RESEARCH_STATE = _RESEARCH_SNAPSHOTS[symbol]
                await _broadcast_research({"type": "snapshot", "data": _RESEARCH_STATE})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("research_worker_error", symbol=symbol, error=str(error))
            _RESEARCH_STATE = {"status": "error", "symbol": symbol, "error": str(error)}
            await _broadcast_research({"type": "status", "data": _RESEARCH_STATE})

        try:
            await asyncio.wait_for(_RESEARCH_WAKE.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass


async def _run_xauusdt_feed() -> None:
    global _LIVE_STATE
    provider = BinanceFuturesProvider()
    while True:
        try:
            end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            start = end - timedelta(days=7)
            bars = await provider.get_bars("XAUUSDT", start, end, timeframe="1h")
            bars = bars.loc[bars.index < pd.Timestamp(end)]
            close = bars["c"].astype(float).tail(24 * 7 + 24)
            if len(close) < 26:
                raise RuntimeError("Not enough closed candles to calculate the XAUUSDT signal.")

            _LIVE_STATE = {
                "history": _candle_payload(close.tail(72)),
                "signal": _signal_payload(close),
            }
            await _broadcast_live({"type": "snapshot", **_LIVE_STATE})

            async for bar in provider.stream_bars("XAUUSDT", timeframe="1h"):
                timestamp = _utc_timestamp(bar.ts)
                if timestamp <= close.index[-1]:
                    continue
                close.loc[timestamp] = float(bar.c)
                close = close.sort_index().tail(24 * 7 + 24)
                candle = {"time": timestamp.isoformat(), "close": float(bar.c)}
                signal = _signal_payload(close)
                history = _candle_payload(close.tail(72))
                _LIVE_STATE = {"history": history, "signal": signal}
                await _broadcast_live({
                    "type": "update",
                    "candle": candle,
                    "signal": signal,
                })
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("xauusdt_feed_error", error=str(error))
            await _broadcast_live({"type": "status", "status": "reconnecting"})
            await asyncio.sleep(5)


async def _run_xauusdt_mark_price() -> None:
    url = f"{BinanceFuturesProvider.BASE}/fapi/v1/premiumIndex"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                symbol = _RESEARCH_SYMBOL
                response = await client.get(url, params={"symbol": symbol})
                response.raise_for_status()
                payload = response.json()
                timestamp = datetime.fromtimestamp(int(payload["time"]) / 1000, timezone.utc)
                _MARK_PRICES[symbol] = float(payload["markPrice"])
                message = {
                    "type": "mark_price",
                    "symbol": symbol,
                    "price": float(payload["markPrice"]),
                    "time": timestamp.isoformat(),
                }
                await _broadcast_live(message)
                await _broadcast_research(message)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log.warning("xauusdt_mark_price_error", error=str(error))
                await _broadcast_live({"type": "mark_status", "status": "reconnecting"})
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("api_start", env=get_settings().env)
    feed_task = asyncio.create_task(_run_xauusdt_feed())
    mark_task = asyncio.create_task(_run_xauusdt_mark_price())
    research_task = asyncio.create_task(_run_research_worker())
    try:
        yield
    finally:
        feed_task.cancel()
        mark_task.cancel()
        research_task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(feed_task, mark_task, research_task)
        log.info("api_stop")


app = FastAPI(title="Quant-Trading-1 API", version="0.1.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-assets")


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


class AssetSelection(BaseModel):
    symbol: str


class PaperOrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    side: str
    quantity: float = Field(gt=0)


def _paper_account() -> PaperTradingAccount:
    global _PAPER_ACCOUNT
    if _PAPER_ACCOUNT is None:
        settings = get_settings()
        _PAPER_ACCOUNT = PaperTradingAccount(
            commission_bps=settings.execution.base_commission_bps,
            slippage_bps=settings.execution.base_slippage_bps,
            max_position_pct=settings.risk.max_position_pct,
            max_gross_leverage=settings.risk.max_gross_leverage,
            max_drawdown=settings.risk.max_drawdown,
        )
    return _PAPER_ACCOUNT


def _paper_price(symbol: str) -> float:
    symbol = symbol.strip().upper()
    price = _MARK_PRICES.get(symbol)
    if price is None and _RESEARCH_STATE.get("symbol") == symbol:
        price = _RESEARCH_STATE.get("price")
    if price is None:
        raise HTTPException(status_code=409, detail="No current market price is available for this symbol.")
    return float(price)


@app.get("/api/research/assets")
async def research_assets():
    try:
        assets = await _fetch_research_assets()
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not load Binance Futures assets: {error}") from error
    return {"assets": assets, "selected": _RESEARCH_SYMBOL}


@app.get("/api/research/state")
async def research_state():
    return _RESEARCH_STATE


@app.post("/api/research/asset")
async def select_research_asset(selection: AssetSelection):
    global _RESEARCH_SYMBOL, _RESEARCH_STATE
    symbol = selection.symbol.strip().upper()
    try:
        assets = await _fetch_research_assets()
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Could not validate Binance Futures asset: {error}") from error
    if symbol not in assets:
        raise HTTPException(status_code=422, detail="Select a currently trading USDT perpetual from Binance Futures.")
    if symbol != _RESEARCH_SYMBOL:
        _RESEARCH_SYMBOL = symbol
        _RESEARCH_STATE = {"status": "connecting", "symbol": symbol}
        _RESEARCH_WAKE.set()
        await _broadcast_research({"type": "status", "data": _RESEARCH_STATE})
    return _RESEARCH_STATE


@app.get("/api/project/modules")
async def project_modules():
    return PROJECT_MODULES


@app.get("/api/project/files")
async def project_files():
    files = []
    for root_name in sorted(SOURCE_ROOTS):
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(PROJECT_ROOT).parts):
                continue
            if path.suffix.lower() in SOURCE_SUFFIXES or path.name == "Dockerfile":
                files.append(path.relative_to(PROJECT_ROOT).as_posix())
    for name in ROOT_SOURCE_FILES:
        if (PROJECT_ROOT / name).is_file():
            files.append(name)
    return sorted(files)


@app.get("/api/project/files/{source_path:path}", response_class=PlainTextResponse)
async def project_file(source_path: str):
    relative = Path(source_path)
    if any(part.startswith(".") for part in relative.parts):
        raise HTTPException(status_code=404, detail="File not found")
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    allowed_root = relative.parts[0] in SOURCE_ROOTS if relative.parts else False
    allowed_root = allowed_root or relative.as_posix() in ROOT_SOURCE_FILES
    if not allowed_root or (resolved.suffix.lower() not in SOURCE_SUFFIXES and resolved.name != "Dockerfile"):
        raise HTTPException(status_code=404, detail="File not found")
    return PlainTextResponse(resolved.read_text(encoding="utf-8"))


@app.get("/risk/limits")
async def risk_limits():
    s = get_settings().risk
    return s.model_dump()


@app.get("/api/paper/account")
async def paper_account():
    return _paper_account().snapshot()


@app.post("/api/paper/order")
async def paper_order(order: PaperOrderRequest):
    symbol = order.symbol.strip().upper()
    try:
        return _paper_account().execute_market(
            symbol, order.side, order.quantity, _paper_price(symbol)
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/paper/close/{symbol}")
async def paper_close(symbol: str):
    symbol = symbol.strip().upper()
    try:
        return _paper_account().close(symbol, _paper_price(symbol))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/paper/reset")
async def paper_reset():
    return _paper_account().reset()


_WS_CLIENTS: set[WebSocket] = set()


@app.websocket("/ws/portfolio")
async def ws_portfolio(ws: WebSocket):
    await ws.accept()
    _WS_CLIENTS.add(ws)
    try:
        while True:
            await asyncio.sleep(1)
            await ws.send_json({"event": "heartbeat"})
    except WebSocketDisconnect:
        _WS_CLIENTS.discard(ws)


@app.websocket("/ws/xauusdt")
async def ws_xauusdt(ws: WebSocket):
    await ws.accept()
    _LIVE_CLIENTS.add(ws)
    try:
        if _LIVE_STATE is None:
            await ws.send_json({"type": "status", "status": "connecting"})
        else:
            await ws.send_json({"type": "snapshot", **_LIVE_STATE})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _LIVE_CLIENTS.discard(ws)


@app.websocket("/ws/research")
async def ws_research(ws: WebSocket):
    await ws.accept()
    _RESEARCH_CLIENTS.add(ws)
    try:
        await ws.send_json({"type": "snapshot", "data": _RESEARCH_STATE})
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _RESEARCH_CLIENTS.discard(ws)


async def broadcast(event: dict) -> None:
    for ws in list(_WS_CLIENTS):
        try:
            await ws.send_json(event)
        except Exception:
            _WS_CLIENTS.discard(ws)
