# Crypto Trading Agent

A Python-based crypto trading assistant powered by Gemini AI and CCXT, providing paper-trading simulation, market analysis, and risk management tools for cryptocurrency trading education and experimentation.

## Features

- **Market Data Integration**: Fetch OHLCV candles from multiple exchanges (Binance, Kraken, Coinbase, KuCoin, Bitfinex) via CCXT with automatic failover.
- **Technical Indicators**: Compute RSI (Relative Strength Index) and other basic indicators for market analysis.
- **Paper Trading Simulator**: Simulate spot trading with full position management including stop-loss and take-profit orders.
- **AI-Powered Assistant**: Unified AI agent (Gemini 2.5 Flash) that interprets market data and executes trading strategies.
- **Risk Management**: Built-in safety caps and risk-based position sizing tools.
- **Portfolio Management**: Track equity, PnL, and trade history with real-time position evaluation.
- **Multi-Timeframe Analysis**: Compare RSI and trends across multiple timeframes (15m, 1h, 4h, etc.).

## Project Structure

```
crypto_trading_agent/
├── README.md                          # Project documentation
├── LICENSE                            # Project license (custom restrictive)
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (not tracked)
├── crypto_trading_agent/
│   ├── __init__.py                   # Package initialization
│   ├── agent.py                      # Unified AI agent and tool wrappers
│   ├── market_tools.py               # Market data and indicator utilities
│   └── simulation.py                 # Paper trading simulator and helpers
└── trade_venv/                       # Python virtual environment
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the repository:**
   ```bash
   cd crypto_trading_agent
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv trade_venv
   .\trade_venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Running with ADK (Windows & Linux)

You can run the project locally using the ADK CLI. Below are example commands for
both Windows (PowerShell) and Linux (bash). Ensure you have installed the
dependencies (`pip install -r requirements.txt`) and activated your virtual
environment so the `adk` command is available on your PATH.

Windows (PowerShell):

```powershell
# activate virtual environment (PowerShell)
# If you see a security error about running scripts being disabled,
# run the following in the same PowerShell session to allow this run only:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\trade_venv\Scripts\Activate.ps1

# Alternatives if you're using a different shell on Windows:
# - CMD (cmd.exe): .\trade_venv\Scripts\activate.bat
# - Git Bash / WSL: source ./trade_venv/Scripts/activate

# run the ADK project
adk run .\crypto_trading_agent\

# start the ADK web UI on port 8000
adk web --port 8000
```

Linux / macOS (bash):

```bash
# create/activate virtual environment (if needed)
python3 -m venv trade_venv
source trade_venv/bin/activate

# run the ADK project
adk run ./crypto_trading_agent/

# start the ADK web UI on port 8000
adk web --port 8000
```

Notes & troubleshooting:
- If `adk` is not found, ensure you installed `google-adk` (or the ADK package)
   into the active environment and that the environment is activated.
- If the web UI does not appear, check firewall/port settings and try a different
   `--port` value.
- If the agent requires cloud LLM keys, create a `.env` with required values
   (for example `GOOGLE_API_KEY`) before running.

### Basic Example

```python
from crypto_trading_agent.agent import root_agent

# Use the agent to analyze BTC/USDT
response = root_agent.run("Fetch 1h BTC/USDT and compute RSI")
print(response)

# Get portfolio state
portfolio = root_agent.run("What is my current portfolio status?")
print(portfolio)

# Simulate a trade
trade_result = root_agent.run("Simulate a 1% risk long position on BTC/USDT with stop at 40000")
print(trade_result)
```

### Core Tools Available

- **`get_ohlcv(pair, timeframe, limit)`**: Fetch OHLCV candles from exchanges.
- **`compute_basic_indicators(candles, rsi_period, pair, timeframe)`**: Compute RSI and other indicators.
- **`sim_place_order(...)`**: Open a simulated long/short position.
- **`sim_close_position(position_id, price)`**: Close an open position.
- **`sim_portfolio_state()`**: Get current portfolio summary.
- **`sim_trade_history(limit)`**: View recent trades and statistics.
- **`sim_reset(initial_balance)`**: Reset the simulator.
- **`eval_strategy_quality(limit)`**: Analyze trade performance metrics.
- **`suggest_notional_from_risk(risk_percent, pair, stop_loss)`**: Size positions based on risk %.
- **`explain_current_exposure()`**: Summarize current risk exposure.

## Configuration

### Exchange Preferences

Customize the fallback exchange order:

```python
from crypto_trading_agent.market_tools import set_fallback_exchanges

set_fallback_exchanges(['kraken', 'binance', 'kucoin'])
```

### Simulator Settings

Adjust the initial balance and risk cap:

```python
from crypto_trading_agent.simulation import GLOBAL_SIMULATOR

GLOBAL_SIMULATOR.reset(initial_balance=50_000.0)
```

## Code Style

This project follows **PEP 8** standards for Python code quality:
- 4-space indentation
- snake_case for functions and variables
- PascalCase for classes
- Consistent import ordering (stdlib → third-party → local)
- NumPy-style docstrings for functions and classes

## Testing

Run syntax and import checks:

```bash
python -m py_compile crypto_trading_agent/agent.py
python -m py_compile crypto_trading_agent/market_tools.py
python -m py_compile crypto_trading_agent/simulation.py
```

## Key Concepts

### Paper Trading
All simulated trades operate in-memory using the `TradingSimulator` class. No real funds or exchanges are involved.

### Risk Management
- Max notional per trade: 20% of equity by default.
- Risk-based sizing: Use `suggest_notional_from_risk()` to size positions based on stop-loss levels.
- Automatic SL/TP evaluation: Positions are closed automatically if stop-loss or take-profit levels are hit.

### Multi-Exchange Failover
If the primary exchange fails, the agent automatically tries fallback exchanges in configured order, ensuring data availability.

## Dependencies

Core dependencies (see `requirements.txt`):
- **ccxt**: Cryptocurrency exchange API abstraction
- **python-dotenv**: Environment variable management
- **google-cloud-aiplatform**: Google AI Platform integration (Gemini)
- **authlib**: Authentication and authorization
- **pydantic**: Data validation and settings management

## Performance & Limitations

- RSI computation uses a simplified algorithm for demo purposes (not a production technical analysis library).
- Simulator assumes instant fills at market price (no slippage or latency modeling).
- Limited to single-market analysis per query (e.g., one pair per `get_ohlcv` call).

## Future Enhancements

- A more capable LLM model
- Scaling up from a single agent to multi-agent
- A database for conversations and system logs
- An agent that monitors logs and the history of ADK activity  

## Educational Disclaimer

This project is designed for **educational and experimentation purposes only**. All trades are simulated and do not execute real transactions. Users should understand that:

- Cryptocurrency markets are volatile and risky.
- Paper trading results do not guarantee real-world performance.
- Past performance does not indicate future results.
- This tool is not a substitute for financial advice.

## License

This project is distributed under two licenses:

### Custom Restricted License
You may:
- ✔ Use the software for personal, educational, and experimental purposes  
- ✔ Run and test the software  
- ✔ View the source code  
- ✔ Modify or create derivative works **only inside your local copy / the same repository**

You may NOT:
- ✘ Redistribute or publish original or modified versions  
- ✘ Remove copyright or license notices  

### Apache License 2.0
This project also includes components provided under the Apache License, Version 2.0.

See the [LICENSE](./LICENSE) file for the full text of both licenses.

## Support

For issues, questions, or suggestions, please open an issue on this repository.

---

**Disclaimer**: This tool is for educational purposes. Use at your own risk and never trade with funds you cannot afford to lose.
