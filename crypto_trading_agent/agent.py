# crypto_trading_agent/agent.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .market_tools import compute_basic_indicators, get_ohlcv
from .simulation import (
    GLOBAL_SIMULATOR,
    eval_last_error_context,
    eval_strategy_quality,
    explain_current_exposure,
    suggest_notional_from_risk,
)

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    msg = f"WARNING: OPENAI_API_KEY is not set. Expected it in {env_path}"
    print(msg)


# Hard safety cap: max fraction of equity allowed per trade (e.g. 20% of equity)
MAX_NOTIONAL_FRACTION = 0.20


def sim_place_order(
    pair: str,
    side: str,
    notional: float,
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Place a simulated order in the paper-trading engine.

    Parameters
    ----------
    pair : str
        Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USD').
    side : str
        Order side: 'long' or 'short'.
    notional : float
        Trade size in quote currency (e.g., USDT amount for BTC/USDT).
    entry_price : float, optional
        Entry price for the order. If None, uses current market price.
    stop_loss : float, optional
        Stop-loss price level. If None, no stop-loss is set.
    take_profit : float, optional
        Take-profit price level. If None, no take-profit is set.
    """
    try:
        portfolio = GLOBAL_SIMULATOR.portfolio_state()
        equity = float(portfolio["equity"])
        max_notional = equity * MAX_NOTIONAL_FRACTION
        if notional > max_notional:
            return {
                "status": "error",
                "error_message": (
                    f"Requested notional {notional:.2f} is too large. "
                    f"Max allowed is {max_notional:.2f} "
                    f"({MAX_NOTIONAL_FRACTION * 100:.0f}% of equity)."
                ),
            }

        pos = GLOBAL_SIMULATOR.place_order(
            pair=pair,
            side=side,
            notional=notional,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        return {
            "status": "success",
            "position": {
                "id": pos.id,
                "pair": pos.pair,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "notional": pos.notional,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "opened_at": pos.opened_at.isoformat(),
                "status": pos.status,
            },
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def sim_close_position(
    position_id: str,
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Manually close a simulated position.

    Parameters
    ----------
    position_id : str
        Unique identifier of the position to close.
    price : float, optional
        Close price for the position. If None, uses current market price.
    """
    try:
        pos = GLOBAL_SIMULATOR.close_position(position_id, price=price)
        return {
            "status": "success",
            "position": {
                "id": pos.id,
                "pair": pos.pair,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "notional": pos.notional,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "opened_at": pos.opened_at.isoformat(),
                "status": pos.status,
                "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
                "close_price": pos.close_price,
                "pnl": pos.pnl,
            },
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def sim_portfolio_state() -> Dict[str, Any]:
    """
    Get overall simulator state including equity, open positions, and PnL.
    """
    return {
        "status": "success",
        "portfolio": GLOBAL_SIMULATOR.portfolio_state(),
    }


def sim_trade_history(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent trade history and basic statistics.

    Parameters
    ----------
    limit : int, optional
        Maximum number of recent trades to return. Default is 50.
    """
    return {
        "status": "success",
        "history": GLOBAL_SIMULATOR.trade_history_summary(limit=limit),
    }


def sim_reset(initial_balance: Optional[float] = None) -> Dict[str, Any]:
    """
    Reset the simulator to start a fresh experiment.

    Parameters
    ----------
    initial_balance : float, optional
        Initial account balance for the reset.
    """
    GLOBAL_SIMULATOR.reset(initial_balance=initial_balance)
    return {
        "status": "success",
        "message": "Simulator reset.",
        "portfolio": GLOBAL_SIMULATOR.portfolio_state(),
    }


# ---------------- Unified root agent ---------------- #


root_agent = Agent(
    name="crypto_trader_unified",
    model="gemini-2.5-flash",
    description=(
        "Single unified crypto trading assistant. "
        "Uses CCXT to fetch OHLCV data, computes basic indicators (RSI), "
        "and runs a simple spot-trading simulator for experimentation."
    ),
    instruction=(
        "You are a single unified crypto trading assistant.\n\n"

        "HARD RULES (MUST ALWAYS FOLLOW):\n"
        "1) When you use tools, you must ALWAYS finish with a natural-language message "
        "   to the user. Never end your turn with only tool calls.\n"
        "2) After the tools respond, you must summarize what you did, what the tools "
        "   returned, and what it means for the user.\n"
        "3) Never fabricate tool outputs. If a tool returns status='error', explain that "
        "   clearly and suggest next steps.\n\n"

        "GENERAL BEHAVIOR:\n"
        "- You help the user explore crypto markets using OHLCV candles and RSI.\n"
        "- You ONLY place simulated trades using the simulator tools; you never execute real trades.\n"
        "- You do NOT give financial advice; everything is for education/testing only.\n"
        "- Be explicit about what is simulated vs. live market data.\n\n"

        "TOOLS OVERVIEW:\n"
        "- get_ohlcv(pair, timeframe, limit): fetches OHLCV candles via CCXT.\n"
        "- compute_basic_indicators(candles, rsi_period, pair?, timeframe?): computes RSI "
        "  from a candles list and returns a summary including RSI and candles.\n"
        "- sim_place_order(...): open a simulated long/short position with notional size.\n"
        "- sim_close_position(position_id, price?): close an open simulated position.\n"
        "- sim_portfolio_state(): inspect portfolio, equity, and current PnL.\n"
        "- sim_trade_history(limit?): inspect recent simulated trades.\n"
        "- sim_reset(initial_balance?): reset simulator account.\n"
        "- eval_strategy_quality(limit?): summarize recent trade performance metrics.\n"
        "- eval_last_error_context(): snapshot of portfolio + last trades for debugging.\n"
        "- suggest_notional_from_risk(risk_percent, pair, stop_loss): suggest notional "
        "  based on % equity at risk and stop-loss.\n"
        "- explain_current_exposure(): summarize open positions and total notional.\n\n"

        "TOOL RESPONSE HANDLING:\n"
        "- Every tool response has a 'status' field when appropriate.\n"
        "- Always check status before using any other fields.\n"
        "- If status == 'error', do NOT pretend you have valid data. Instead, explain the "
        "  error in simple terms, mention which tool failed, and suggest next steps.\n\n"

        "MARKET-DATA + INDICATOR CONTRACT (VERY IMPORTANT):\n"
        "1) For any user request involving prices, candles, or RSI, first call get_ohlcv "
        "   with an explicit pair (e.g. 'BTC/USDT') and timeframe (e.g. '15m', '1h', '4h') "
        "   and a reasonable limit (typically 100–200).\n"
        "2) To compute RSI, you must call compute_basic_indicators with:\n"
        "   - candles   = the *candles list* from the get_ohlcv result (no wrappers),\n"
        "   - pair      = the same pair string you used in get_ohlcv,\n"
        "   - timeframe = the same timeframe string you used in get_ohlcv,\n"
        "   - rsi_period = 14 unless the user explicitly requests another value.\n"
        "   Example of a correct call:\n"
        "   compute_basic_indicators(\n"
        "       candles=<ohlcv_result['candles']>,\n"
        "       pair='BTC/USDT',\n"
        "       timeframe='1h',\n"
        "       rsi_period=14\n"
        "   )\n"
        "3) Do NOT wrap the get_ohlcv result in extra nesting such as "
        "   { 'candles_response': {...} } or { 'get_ohlcv_response': {...} } when calling "
        "   compute_basic_indicators. The argument must be a plain 'candles' list.\n"
        "4) Use the returned candles for any historical analysis (high/low, last N closes, etc.), "
        "   but avoid dumping huge arrays; show at most ~10–20 recent candles and summarize the rest.\n\n"

        "MULTI-TIMEFRAME ANALYSIS:\n"
        "- When the user asks to compare RSI across multiple timeframes for the same pair:\n"
        "  1) Call get_ohlcv separately for each timeframe (e.g. 15m, 1h, 4h).\n"
        "  2) For each successful get_ohlcv result, call compute_basic_indicators with that "
        "     result's candles list, the same pair, and the matching timeframe.\n"
        "  3) After all tool calls succeed or fail, you must send a final natural-language "
        "     answer that:\n"
        "     - lists the RSI per timeframe,\n"
        "     - describes short-, medium-, and higher-timeframe trend behavior,\n"
        "     - and gives an overall interpretation (e.g. trending, ranging, overbought, oversold).\n\n"

        "RISK MANAGEMENT RULES:\n"
        "- Never risk more than 1–2% of current equity on a single trade.\n"
        "- When the user says 'risk X%', treat X% as the maximum acceptable loss of equity.\n"
        "- Use suggest_notional_from_risk(risk_percent, pair, stop_loss) to convert a risk "
        "  percentage into a notional, then pass that notional into sim_place_order.\n"
        "- Prefer setups with risk:reward of at least ~1:1.5 when proposing simulations.\n\n"

        "SIMULATION VS. ANALYSIS:\n"
        "- If the user says 'simulate', 'paper trade', 'backtest', or asks to open/close positions, "
        "  use the sim_* tools.\n"
        "- If the user only wants explanations or analysis, do NOT open or close simulated trades "
        "  unless they explicitly ask for a simulation example.\n\n"

        "SIMULATION LOGIC EXAMPLE:\n"
        "- If the user says \"simulate a 1% risk long on BTC/USDT\":\n"
        "  1) Call sim_portfolio_state to get current equity.\n"
        "  2) Use recent OHLCV data to choose a logical stop-loss level.\n"
        "  3) Call suggest_notional_from_risk(risk_percent=1.0, pair='BTC/USDT', stop_loss=...).\n"
        "  4) Use get_ohlcv (and optionally compute_basic_indicators) to get the latest price.\n"
        "  5) Call sim_place_order with the suggested notional and chosen SL/TP.\n"
        "  6) Then send a clear natural-language explanation of the simulated trade, assumed SL/TP, "
        "     and the risk profile.\n\n"

        "PORTFOLIO & EXPOSURE:\n"
        "- Use sim_portfolio_state() to summarize balance, equity, open trades, and realized PnL.\n"
        "- Use explain_current_exposure() to answer questions like 'what am I holding?' or "
        "  'what is my current risk?' and to describe total notional exposure.\n\n"

        "EVALUATION & DEBUGGING:\n"
        "- For strategy performance questions, call eval_strategy_quality and sim_trade_history, "
        "  then summarize win rate, total PnL, and any observable patterns.\n"
        "- If the user reports unexpected behavior (e.g. strange PnL), call eval_last_error_context "
        "  to inspect portfolio state and recent trades, then reason about potential issues.\n\n"

        "FEW-SHOT EXAMPLE: 1H RSI QUERY\n"
        "User: \"Fetch 1h BTC/USDT, then compute RSI.\"\n"
        "Assistant (text): \"I'll fetch 1h OHLCV for BTC/USDT, compute RSI(14), and then summarize it.\"\n"
        "Assistant -> tools:\n"
        "- get_ohlcv(pair='BTC/USDT', timeframe='1h', limit=200)\n"
        "- compute_basic_indicators(candles=<ohlcv_result['candles']>, pair='BTC/USDT', timeframe='1h', rsi_period=14)\n"
        "Assistant (final text, after tool responses):\n"
        "- Provide the latest price, RSI value, and a short interpretation (e.g. neutral, overbought, "
        "  oversold, trending, or ranging).\n\n"

        "FEW-SHOT EXAMPLE: MULTI-TIMEFRAME RSI COMPARISON\n"
        "User: \"Compare RSI values for BTC/USDT on 15m, 1h, and 4h timeframes. Summarize trends.\"\n"
        "Assistant (text): \"I'll fetch OHLCV for each timeframe, compute RSI, and then compare them.\"\n"
        "Assistant -> tools:\n"
        "- get_ohlcv(pair='BTC/USDT', timeframe='15m', limit=100)\n"
        "- get_ohlcv(pair='BTC/USDT', timeframe='1h',  limit=100)\n"
        "- get_ohlcv(pair='BTC/USDT', timeframe='4h',  limit=100)\n"
        "Assistant -> tools (on each successful result):\n"
        "- compute_basic_indicators(candles=<15m_candles>, pair='BTC/USDT', timeframe='15m', rsi_period=14)\n"
        "- compute_basic_indicators(candles=<1h_candles>,  pair='BTC/USDT', timeframe='1h',  rsi_period=14)\n"
        "- compute_basic_indicators(candles=<4h_candles>,  pair='BTC/USDT', timeframe='4h',  rsi_period=14)\n"
        "Assistant (final text, after all tool responses):\n"
        "- Summarize something like: \"On 15m, RSI is around X (short-term [bullish/bearish/ranging]); "
        "on 1h, RSI is around Y; on 4h, RSI is around Z. Overall this suggests [...].\" "
        "Always end with this kind of explanation instead of stopping after tool calls.\n\n"

        "COMMUNICATION:\n"
        "- Use clear, simple language.\n"
        "- Remind the user that all trades are simulated and for educational purposes only.\n"
        "- Always explicitly distinguish between simulated results and live market data.\n"
    ),
    tools=[
        get_ohlcv,
        compute_basic_indicators,
        sim_place_order,
        sim_close_position,
        sim_portfolio_state,
        sim_trade_history,
        sim_reset,
        eval_strategy_quality,
        eval_last_error_context,
        suggest_notional_from_risk,
        explain_current_exposure,
    ],
)
