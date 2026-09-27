"""In-memory paper trading account for the web research dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from threading import RLock


@dataclass
class PaperPosition:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    last_price: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.last_price)

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * (self.last_price - self.average_price)


@dataclass
class PaperTradingAccount:
    initial_cash: float = 10_000.0
    commission_bps: float = 0.5
    slippage_bps: float = 0.8
    max_position_pct: float = 0.25
    max_gross_leverage: float = 3.0
    max_drawdown: float = 0.15
    cash: float = field(init=False)
    realized_pnl: float = field(default=0.0, init=False)
    fees_paid: float = field(default=0.0, init=False)
    peak_equity: float = field(init=False)
    max_drawdown_seen: float = field(default=0.0, init=False)
    killed: bool = field(default=False, init=False)
    positions: dict[str, PaperPosition] = field(default_factory=dict, init=False)
    trades: list[dict] = field(default_factory=list, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        self.cash = float(self.initial_cash)
        self.peak_equity = float(self.initial_cash)

    def reset(self) -> dict:
        with self._lock:
            self.cash = float(self.initial_cash)
            self.realized_pnl = 0.0
            self.fees_paid = 0.0
            self.peak_equity = float(self.initial_cash)
            self.max_drawdown_seen = 0.0
            self.killed = False
            self.positions.clear()
            self.trades.clear()
            return self.snapshot()

    def mark(self, symbol: str, price: float) -> dict:
        with self._lock:
            self._validate_price(price)
            position = self.positions.get(symbol)
            if position:
                position.last_price = price
                self._update_risk_state()
            return self.snapshot()

    def execute_market(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        with self._lock:
            symbol = symbol.strip().upper()
            side = side.strip().lower()
            self._validate_price(price)
            if side not in {"buy", "sell"}:
                raise ValueError("Side must be buy or sell")
            if not isfinite(quantity) or quantity <= 0:
                raise ValueError("Quantity must be positive")
            if self.killed:
                raise ValueError("Account is halted by the drawdown kill switch")

            position = self.positions.setdefault(symbol, PaperPosition(symbol=symbol))
            position.last_price = price
            signed_delta = quantity if side == "buy" else -quantity
            target_quantity = position.quantity + signed_delta
            equity_before = self._equity()
            gross_after = abs(target_quantity * price) + sum(
                item.notional for key, item in self.positions.items() if key != symbol
            )
            if gross_after > equity_before * self.max_gross_leverage + 1e-9:
                raise ValueError("Order exceeds maximum gross leverage")
            if abs(target_quantity * price) > equity_before * self.max_position_pct + 1e-9:
                raise ValueError("Order exceeds maximum position size")

            fill_price = price * (1 + self.slippage_bps * 1e-4 * (1 if side == "buy" else -1))
            fee = abs(quantity * fill_price) * self.commission_bps * 1e-4
            realized = self._apply_fill(position, signed_delta, fill_price)
            self.realized_pnl += realized
            self.cash += realized - fee
            self.fees_paid += fee
            position.last_price = price
            if abs(position.quantity) < 1e-12:
                self.positions.pop(symbol, None)
            self._update_risk_state()
            trade = {
                "id": len(self.trades) + 1,
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": fill_price,
                "fee": fee,
                "realized_pnl": realized,
            }
            self.trades.append(trade)
            return self.snapshot()

    def close(self, symbol: str, price: float) -> dict:
        with self._lock:
            position = self.positions.get(symbol.upper())
            if not position or position.quantity == 0:
                raise ValueError("No open position for this symbol")
            side = "sell" if position.quantity > 0 else "buy"
            return self.execute_market(symbol, side, abs(position.quantity), price)

    def snapshot(self) -> dict:
        equity = self._equity()
        return {
            "status": "PAPER",
            "currency": "USDT",
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "equity": equity,
            "net_return": equity / self.initial_cash - 1.0,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(item.unrealized_pnl for item in self.positions.values()),
            "fees_paid": self.fees_paid,
            "max_drawdown": self.max_drawdown_seen,
            "killed": self.killed,
            "gross_exposure": sum(item.notional for item in self.positions.values()),
            "positions": [
                {
                    "symbol": item.symbol,
                    "quantity": item.quantity,
                    "average_price": item.average_price,
                    "last_price": item.last_price,
                    "notional": item.notional,
                    "unrealized_pnl": item.unrealized_pnl,
                }
                for item in self.positions.values()
            ],
            "trades": self.trades[-50:],
        }

    @staticmethod
    def _validate_price(price: float) -> None:
        if not isfinite(price) or price <= 0:
            raise ValueError("Price must be finite and positive")

    def _apply_fill(self, position: PaperPosition, delta: float, fill_price: float) -> float:
        old = position.quantity
        new = old + delta
        realized = 0.0
        if old and old * delta < 0:
            closed = min(abs(old), abs(delta))
            realized = closed * (fill_price - position.average_price) * (1 if old > 0 else -1)
        if old == 0 or old * delta > 0:
            position.average_price = (
                (abs(old) * position.average_price + abs(delta) * fill_price) / abs(new)
            )
        elif new and old * new < 0:
            position.average_price = fill_price
        position.quantity = new
        return realized

    def _equity(self) -> float:
        return self.cash + sum(item.unrealized_pnl for item in self.positions.values())

    def _update_risk_state(self) -> None:
        equity = self._equity()
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        self.max_drawdown_seen = min(self.max_drawdown_seen, drawdown)
        if drawdown <= -self.max_drawdown:
            self.killed = True
