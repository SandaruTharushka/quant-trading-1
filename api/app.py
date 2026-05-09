"""FastAPI app exposing health, signals, risk, and a WebSocket for live updates."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core.config import get_settings
from core.logging import get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_start", env=get_settings().env)
    yield
    log.info("api_stop")


app = FastAPI(title="Quant-Trading-1 API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/risk/limits")
async def risk_limits():
    s = get_settings().risk
    return s.model_dump()


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


async def broadcast(event: dict) -> None:
    for ws in list(_WS_CLIENTS):
        try:
            await ws.send_json(event)
        except Exception:
            _WS_CLIENTS.discard(ws)
