# crypto_trading_agent/simulation.py

from __future__ import annotations

import datetime
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SimPosition:
    """Dataclass representing a simulated spot trading position.

    Attributes
    ----------
    id : str
        Unique identifier for the position.
    pair : str
        Market symbol, e.g. 'BTC/USDT'.
    side : str
        Either 'long' or 'short'.
    entry_price : float
        Entry price used for the position.
    notional : float
        Amount of quote currency committed (e.g., USDT).
    stop_loss : Optional[float]
        Stop-loss price, if any.
    take_profit : Optional[float]
        Take-profit price, if any.
    opened_at : datetime.datetime
        Timestamp when the position was opened (UTC).
    status : str
        'open' or 'closed'.
    closed_at : Optional[datetime.datetime]
        Timestamp when the position was closed (UTC), if closed.
    close_price : Optional[float]
        Price at which the position was closed.
    pnl : Optional[float]
        Realized PnL for the position in quote currency.
    """
    id: str
    pair: str
    side: str
    entry_price: float
    notional: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime.datetime
    status: str = "open"
    closed_at: Optional[datetime.datetime] = None
    close_price: Optional[float] = None
    pnl: Optional[float] = None


class TradingSimulator:
    """
    Very simple portfolio simulator for spot trading.
    - Uses 'notional' as the amount of quote currency committed per trade.
    - PnL is measured in the same quote currency (e.g. USDT).
    """

    def __init__(self, initial_balance: float = 10_000.0):
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.positions: Dict[str, SimPosition] = {}
        self.trade_history: List[SimPosition] = []
        self.last_prices: Dict[str, float] = {}

    def _now(self) -> datetime.datetime:
        """Return the current UTC datetime.

        Returns
        -------
        datetime.datetime
            Current UTC datetime.
        """
        return datetime.datetime.utcnow()

    def _equity(self) -> float:
        """Compute total equity as cash plus unrealized PnL.

        Returns
        -------
        float
            Total account equity in quote currency.
        """
        unrealized = 0.0
        for pos in self.positions.values():
            price = self.last_prices.get(pos.pair)
            if price is None or pos.status != "open":
                continue
            unrealized += self._calc_pnl_price(pos, price)
        return self.cash + unrealized

    def _calc_pnl_price(self, pos: SimPosition, price: float) -> float:
        """Calculate PnL for a position at a given market `price`.

        Parameters
        ----------
        pos : SimPosition
            Open position to evaluate.
        price : float
            Market price to use for PnL calculation.

        Returns
        -------
        float
            Unrealized PnL in quote currency (can be negative).
        """
        if pos.side == "long":
            return pos.notional * (price / pos.entry_price - 1.0)
        else:
            return pos.notional * (pos.entry_price / price - 1.0)

    def reset(self, initial_balance: Optional[float] = None) -> None:
        """Reset the simulator state.

        Parameters
        ----------
        initial_balance : float, optional
            If provided, set a new initial balance. Otherwise the existing
            `initial_balance` is preserved.
        """
        if initial_balance is not None:
            self.initial_balance = float(initial_balance)
        self.cash = float(self.initial_balance)
        self.positions.clear()
        self.trade_history.clear()
        self.last_prices.clear()

    def update_price(self, pair: str, price: float) -> None:
        """Update the last price for a market and evaluate SL/TP.

        Parameters
        ----------
        pair : str
            Market symbol, e.g. ``'BTC/USDT'``.
        price : float
            Latest market price for the `pair`.
        """
        self.last_prices[pair] = float(price)
        for pos in list(self.positions.values()):
            if pos.pair != pair or pos.status != "open":
                continue
            self._check_and_maybe_close(pos, price)

    def place_order(
        self,
        pair: str,
        side: str,
        notional: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> SimPosition:
        """Place a simulated order and reserve notional cash.

        Parameters
        ----------
        pair : str
            Market symbol to trade.
        side : str
            ``'long'`` or ``'short'``.
        notional : float
            Amount of quote currency to commit to the position.
        entry_price : float, optional
            Price to use as entry. If omitted, the last known price for the
            pair is used (must be present via ``update_price``).
        stop_loss : float, optional
            Stop-loss price.
        take_profit : float, optional
            Take-profit price.

        Returns
        -------
        SimPosition
            The created position object.

        Raises
        ------
        ValueError
            For invalid side, insufficient cash, or missing price.
        """
        side = side.lower()
        if side not in {"long", "short"}:
            raise ValueError("side must be 'long' or 'short'")
        notional = float(notional)
        if notional <= 0:
            raise ValueError("notional must be > 0")

        if notional > self.cash:
            raise ValueError(
                f"Insufficient cash: requested {notional}, available {self.cash}"
            )

        if entry_price is None:
            price = self.last_prices.get(pair)
            if price is None:
                raise ValueError(
                    f"No last price for {pair}. Call update_price or use entry_price."
                )
            entry_price = price

        self.cash -= notional
        pos = SimPosition(
            id=str(uuid.uuid4()),
            pair=pair,
            side=side,
            entry_price=float(entry_price),
            notional=notional,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=self._now(),
        )
        self.positions[pos.id] = pos
        return pos

    def close_position(self, pos_id: str, price: Optional[float] = None) -> SimPosition:
        """Close an open position and realize PnL.

        Parameters
        ----------
        pos_id : str
            Identifier of the position to close.
        price : float, optional
            Exit price. If omitted, uses the last known price for the
            position's market.

        Returns
        -------
        SimPosition
            The closed position (with PnL and close timestamp set).

        Raises
        ------
        KeyError
            If ``pos_id`` is not found.
        ValueError
            If no price is available to close the position.
        """
        if pos_id not in self.positions:
            raise KeyError(f"Position {pos_id} not found")

        pos = self.positions[pos_id]
        if pos.status != "open":
            return pos

        if price is None:
            price = self.last_prices.get(pos.pair)
            if price is None:
                raise ValueError(
                    f"No last price for {pos.pair}. Call update_price or pass price."
                )

        price = float(price)
        pnl = self._calc_pnl_price(pos, price)
        self.cash += pos.notional + pnl

        pos.status = "closed"
        pos.closed_at = self._now()
        pos.close_price = price
        pos.pnl = pnl

        self.trade_history.append(pos)
        del self.positions[pos_id]
        return pos

    def _check_and_maybe_close(self, pos: SimPosition, price: float) -> None:
        """Check a position against SL/TP and close it if a trigger is hit.

        Parameters
        ----------
        pos : SimPosition
            Position to evaluate.
        price : float
            Current market price for the position's pair.
        """
        hit_sl = pos.stop_loss is not None and (
            (pos.side == "long" and price <= pos.stop_loss)
            or (pos.side == "short" and price >= pos.stop_loss)
        )
        hit_tp = pos.take_profit is not None and (
            (pos.side == "long" and price >= pos.take_profit)
            or (pos.side == "short" and price <= pos.take_profit)
        )

        if hit_sl or hit_tp:
            self.close_position(pos.id, price=price)

    def portfolio_state(self) -> dict:
        """Return a dictionary summarizing current portfolio state.

        Returns
        -------
        dict
            Summary including initial balance, cash, equity, open positions,
            realized PnL and trade counts.
        """
        open_positions = [asdict(p) for p in self.positions.values()]
        realized_pnl = sum((p.pnl or 0.0) for p in self.trade_history)
        return {
            "initial_balance": self.initial_balance,
            "cash": self.cash,
            "equity": self._equity(),
            "open_positions": open_positions,
            "realized_pnl": realized_pnl,
            "open_count": len(open_positions),
            "trade_count": len(self.trade_history),
        }

    def trade_history_summary(self, limit: int = 50) -> dict:
        """Return a summary of recent trades.

        Parameters
        ----------
        limit : int, optional
            Maximum number of recent trades to include. Defaults to ``50``.

        Returns
        -------
        dict
            Summary including counts, win rate, total PnL and serialized trades.
        """
        trades = self.trade_history[-limit:]
        wins = sum(1 for t in trades if (t.pnl or 0.0) > 0)
        losses = sum(1 for t in trades if (t.pnl or 0.0) < 0)
        total = len(trades)
        win_rate = wins / total if total > 0 else 0.0
        total_pnl = sum((t.pnl or 0.0) for t in trades)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "trades": [asdict(t) for t in trades],
        }


GLOBAL_SIMULATOR = TradingSimulator(initial_balance=10_000.0)

# Evaluation & debugging helper tools


def eval_strategy_quality(limit: int = 50) -> Dict[str, Any]:
    """Compute simple quality metrics for recent simulated trades.

    Parameters
    ----------
    limit : int, optional
        Number of recent trades to analyze. Defaults to ``50``.

    Returns
    -------
    dict
        Response containing a status, aggregated metrics and raw trades.
    """
    summary = GLOBAL_SIMULATOR.trade_history_summary(limit=limit)
    return {
        "status": "success",
        "metrics": {
            "total_trades": summary["total_trades"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "win_rate": summary["win_rate"],
            "total_pnl": summary["total_pnl"],
        },
        "raw_trades": summary["trades"],
    }


def eval_last_error_context() -> Dict[str, Any]:
    """Provide simple debug context: portfolio state and recent trades.

    Returns
    -------
    dict
        Response including current portfolio summary and recent trade history.
    """
    portfolio = GLOBAL_SIMULATOR.portfolio_state()
    history = GLOBAL_SIMULATOR.trade_history_summary(limit=10)
    return {
        "status": "success",
        "portfolio": portfolio,
        "recent_trades": history,
    }

# Risk & exposure helper tools


def suggest_notional_from_risk(
    risk_percent: float,
    pair: str,
    stop_loss: float,
) -> Dict[str, Any]:
    """
    Suggest a notional size such that the max loss if stop_loss is hit
    is approximately `risk_percent` of current equity.

    Parameters
    ----------
    risk_percent : float
        Percentage of equity to risk (e.g., 1.0 for 1%).
    pair : str
        Trading pair symbol, e.g. 'BTC/USDT'.
    stop_loss : float
        Planned stop-loss price.

    Returns
    -------
    dict
        - status: 'success' or 'error'
        - notional: float (suggested notional, only on success)
        - equity: float (current equity)
        - last_price: float (last known price for the pair)
        - risk_amount: float (risk in quote currency)
        - message: explanation
    """
    try:
        portfolio = GLOBAL_SIMULATOR.portfolio_state()
        equity = float(portfolio["equity"])
        if equity <= 0:
            return {
                "status": "error",
                "error_message": "Equity is non-positive; cannot size position.",
            }

        last_price = GLOBAL_SIMULATOR.last_prices.get(pair)
        if last_price is None:
            return {
                "status": "error",
                "error_message": (
                    f"No last price for {pair}. Fetch candles first so "
                    "update_price() has been called."
                ),
            }

        risk_fraction = float(risk_percent) / 100.0
        if risk_fraction <= 0:
            return {
                "status": "error",
                "error_message": "risk_percent must be > 0.",
            }

        stop_dist = abs(1.0 - (stop_loss / last_price))
        if stop_dist <= 0:
            return {
                "status": "error",
                "error_message": (
                    "Stop distance is zero; choose a different stop_loss."
                ),
            }

        risk_amount = equity * risk_fraction
        notional = risk_amount / stop_dist

        return {
            "status": "success",
            "notional": notional,
            "equity": equity,
            "last_price": last_price,
            "risk_amount": risk_amount,
            "message": (
                f"With equity={equity:.2f}, risk={risk_percent:.2f}% "
                f"and stop at {stop_loss}, suggested notional is {notional:.2f}."
            ),
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def explain_current_exposure() -> Dict[str, Any]:
    """
    High-level summary of current portfolio exposure to help the LM
    explain the situation to the user.
    """
    portfolio = GLOBAL_SIMULATOR.portfolio_state()
    open_positions = portfolio["open_positions"]
    total_notional_committed = sum(p["notional"] for p in open_positions)

    return {
        "status": "success",
        "portfolio": portfolio,
        "exposure": {
            "open_positions_count": portfolio["open_count"],
            "total_notional_committed": total_notional_committed,
        },
    }
