# crypto_trading_agent/market_tools.py

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import ccxt

from .simulation import GLOBAL_SIMULATOR


_FALLBACK_EXCHANGES = [
    "binance",
    "kraken",
    "coinbaseexchange",
    "kucoin",
    "bitfinex",
]


def _instantiate_exchange(exchange_id: str):
    """Create a ccxt exchange instance by id with sensible defaults.

    Parameters
    ----------
    exchange_id : str
        Exchange identifier as used by CCXT (lowercase), e.g. 'binance'.

    Returns
    -------
    Exchange
        An instantiated CCXT exchange object with `enableRateLimit=True`.

    Raises
    ------
    ValueError
        If the provided `exchange_id` is not supported by the installed CCXT
        library.
    """
    if exchange_id not in ccxt.exchanges:
        raise ValueError(f"Exchange id not supported by ccxt: {exchange_id}")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"enableRateLimit": True})


_primary_exchange_id = _FALLBACK_EXCHANGES[0]
_exchange = _instantiate_exchange(_primary_exchange_id)
_exchange_instances: Dict[str, Any] = {_primary_exchange_id: _exchange}


def set_fallback_exchanges(order: list[str]) -> None:
    """Set the preferred exchange order for failover attempts.

    Parameters
    ----------
    order : list[str]
        List of exchange ids (lowercase) in the order they'd be attempted.

    Examples
    --------
    >>> set_fallback_exchanges(['kraken', 'binance', 'kucoin'])
    """
    global _FALLBACK_EXCHANGES
    if not order:
        raise ValueError("Fallback exchange list must not be empty")
    _FALLBACK_EXCHANGES = [e.lower() for e in order]


def set_primary_exchange(exchange_id: str) -> None:
    """Set the primary exchange used for requests (instantiates it).

    Parameters
    ----------
    exchange_id : str
        Exchange id (lowercase) to use as the primary exchange.
    """
    global _primary_exchange_id, _exchange
    _primary_exchange_id = exchange_id.lower()
    _exchange = _instantiate_exchange(_primary_exchange_id)


def get_ohlcv(
    pair: str,
    timeframe: str = "1h",
    limit: int = 200,
) -> Dict[str, Any]:
    """Fetch OHLCV candles for a symbol and timeframe using CCXT.

    Parameters
    ----------
    pair : str
        Market symbol, for example ``'BTC/USDT'``.
    timeframe : str, optional
        Timeframe for OHLCV candles, e.g. ``'1m'``, ``'1h'``, ``'1d'``.
        Defaults to ``'1h'``.
    limit : int, optional
        Number of candles to return. Defaults to ``200``.

    Returns
    -------
    dict
        A response dictionary with the following keys on success:
        - ``status``: ``'success'``
        - ``pair``: requested market symbol
        - ``timeframe``: requested timeframe
        - ``exchange``: the exchange id that supplied the data
        - ``candles``: list of candle dicts with keys
          ``timestamp, time_iso, open, high, low, close, volume``

        On failure returns a dict with ``status: 'error'`` and
        ``error_message`` describing attempts made.
    """
    global _primary_exchange_id, _exchange

    MIN_CANDLES = 60
    if limit < MIN_CANDLES:
        limit = MIN_CANDLES

    candidates = [
        _primary_exchange_id,
    ] + [e for e in _FALLBACK_EXCHANGES if e != _primary_exchange_id]

    raw = None
    last_error_messages: List[str] = []
    used_exchange_id: str | None = None

    for ex_id in candidates:
        try:
            if ex_id in _exchange_instances:
                ex = _exchange_instances[ex_id]
            else:
                ex = _instantiate_exchange(ex_id)
                _exchange_instances[ex_id] = ex

            timeframes = getattr(ex, "timeframes", None)
            if timeframes is not None and timeframe not in timeframes:
                last_error_messages.append(
                    f"{ex_id}: timeframe '{timeframe}' not supported"
                )
                continue

            raw = ex.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            used_exchange_id = ex_id
            _primary_exchange_id = ex_id
            _exchange = ex
            break
        except Exception as e:
            last_error_messages.append(f"{ex_id}: {type(e).__name__}: {e}")
            raw = None

    if raw is None:
        return {
            "status": "error",
            "error_message": (
                "Failed to fetch OHLCV from all candidate exchanges. "
                f"Attempts: {last_error_messages}"
            ),
        }

    candles: List[Dict[str, Any]] = []
    for ts, o, h, l, c, v in raw:
        candles.append(
            {
                "timestamp": int(ts),
                "time_iso": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts / 1000)
                ),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )

    if candles:
        last_price = candles[-1]["close"]
        GLOBAL_SIMULATOR.update_price(pair, last_price)

    return {
        "status": "success",
        "pair": pair,
        "timeframe": timeframe,
        "exchange": used_exchange_id,
        "candles": candles,
    }

# RSI + indicator summary helpers


def simple_rsi(prices: list[float], period: int = 14) -> float:
    """Compute a simple Relative Strength Index (RSI) value.

    This is a minimal RSI implementation intended for demo/testing only.

    Parameters
    ----------
    prices : list[float]
        Sequence of historical closing prices (oldest first).
    period : int, optional
        Lookback period for RSI. Defaults to ``14``.

    Returns
    -------
    float
        RSI value in the range ``0.0`` to ``100.0``.

    Raises
    ------
    ValueError
        If the provided ``prices`` list is shorter than ``period + 1``.
    """
    if len(prices) < period + 1:
        raise ValueError("Not enough data for RSI")

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(-diff)

    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_basic_indicators(
    candles: List[Dict[str, Any]],
    rsi_period: int = 14,
    pair: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute basic indicators (currently RSI) from OHLCV candles.

    Parameters
    ----------
    candles : list[dict]
        A list of candle dictionaries, each with keys
        'timestamp', 'time_iso', 'open', 'high', 'low', 'close', 'volume'.
        This should usually be the `candles` list returned by `get_ohlcv`.
    rsi_period : int, optional
        Lookback period for RSI. Defaults to 14.
    pair : str, optional
        Market symbol (e.g. 'BTC/USDT'). If omitted, 'UNKNOWN' is used.
    timeframe : str, optional
        Timeframe string (e.g. '1h', '15m'). If omitted, 'UNKNOWN' is used.

    Returns
    -------
    dict
        On success:
        - status: "success"
        - pair, timeframe
        - latest_price, latest_time
        - rsi
        - meta: { candle_count, rsi_period }
        - candles: the same candles list
        On error:
        - status: "error"
        - error_message: description of the problem.
    """

    if not isinstance(candles, list) or len(candles) == 0:
        return {
            "status": "error",
            "error_message": (
                "compute_basic_indicators expected a non-empty list for 'candles'."
            ),
        }

    try:
        closes = [float(c["close"]) for c in candles]
    except Exception as e:
        return {
            "status": "error",
            "error_message": (
                f"Failed to extract 'close' prices from candles: {type(e).__name__}: {e}"
            ),
        }

    if len(closes) < rsi_period + 1:
        return {
            "status": "error",
            "error_message": (
                f"Not enough candles to compute RSI: got {len(closes)}, "
                f"need at least {rsi_period + 1}."
            ),
        }

    rsi_value = simple_rsi(closes, period=rsi_period)
    latest = candles[-1]

    return {
        "status": "success",
        "pair": pair or "UNKNOWN",
        "timeframe": timeframe or "UNKNOWN",
        "latest_price": float(latest["close"]),
        "latest_time": latest.get("time_iso"),
        "rsi": rsi_value,
        "meta": {
            "candle_count": len(candles),
            "rsi_period": rsi_period,
        },
        "candles": candles,
    }
