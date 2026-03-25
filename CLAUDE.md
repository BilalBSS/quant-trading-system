# CLAUDE.md — Quant Trading System Blueprint

## Project Overview

Self-improving agentic trading system for US stocks (via Alpaca, commission-free) and crypto (via Alpaca initially, Coinbase later). The system runs 24/7 on a VPS (~$8/month), uses free data APIs, and evolves its own strategies using a Karpathy autoresearch-inspired feedback loop.

**Core principle**: Every strategy gets a fundamental base layer (always present) + a technical signal layer (varies per strategy). Fundamentals are the cornerstone. Technicals are the timing mechanism.

**Budget constraint**: Total infrastructure cost must stay under $25/month. Use free tiers aggressively (Groq free for lightweight LLM, edgartools for SEC data, yfinance for fundamentals, Alpaca free tier for market data, Finnhub for events/sentiment, CoinGecko for crypto market data, DefiLlama for DeFi flows, Dune Analytics for on-chain).

## Progress

- **Phase 1** (foundation): COMPLETE — db, symbols, resilience, validators, migrations
- **Phase 2** (data layer + regime detection): COMPLETE — market_data, fundamentals, sec_filings, regime_detector, backfill script
- **Phase 3** (analysis engine): NOT STARTED
- **173 tests passing** (98 Phase 1 + 75 Phase 2)

## Architecture

```
quant-trading-system/
├── CLAUDE.md                    # this file — project blueprint
├── CHANGELOG.md                 # version history
├── TODOS.md                     # deferred work items
├── main.py                      # entry point — starts the agent orchestrator
├── evolve.py                    # autoresearch loop — runs nightly via cron
├── requirements.txt             # python dependencies
├── .env.example                 # environment variables template
├── configs/
│   ├── strategies/              # strategy config files (JSON)
│   │   ├── strategy_001.json    # each file defines entry/exit rules, indicators, params
│   │   └── ...
│   ├── evolution.md             # instructions for the evolution agent
│   └── risk_limits.json         # portfolio-wide risk constraints
├── scripts/
│   ├── __init__.py
│   └── backfill.py              # [BUILT] backfill: market_data -> regimes -> fundamentals -> insider trades
├── src/
│   ├── brokers/                 # modular broker layer — swappable (not yet built)
│   │   ├── base.py              # abstract BrokerInterface
│   │   ├── alpaca_broker.py     # Alpaca implementation (stocks + crypto)
│   │   ├── coinbase_broker.py   # Coinbase implementation (future, crypto)
│   │   ├── paper_broker.py      # simulated broker for backtesting
│   │   └── broker_factory.py    # routes symbols to correct broker
│   ├── data/                    # data ingestion layer — all free sources
│   │   ├── __init__.py
│   │   ├── db.py                # [BUILT] asyncpg pool + migration runner for Neon PostgreSQL
│   │   ├── symbols.py           # [BUILT] symbol normalization (internal <-> Alpaca format)
│   │   ├── resilience.py        # [BUILT] retry + circuit breaker decorator
│   │   ├── validators.py        # [BUILT] data validation bounds checking
│   │   ├── market_data.py       # [BUILT] alpaca ohlcv + yfinance fallback + backfill
│   │   ├── fundamentals.py      # [BUILT] yfinance: P/E, P/S, revenue, margins, sector averages
│   │   ├── sec_filings.py       # [BUILT] edgartools: Form 4 insider trades
│   │   ├── regime_detector.py   # [BUILT] market regime classification (bull/bear/sideways/high_vol)
│   │   ├── news_sentiment.py    # Finnhub events + sentiment layer
│   │   ├── crypto_data.py       # CoinGecko price/market data + DefiLlama flows
│   │   ├── crypto_onchain.py    # Dune Analytics custom on-chain queries
│   │   └── migrations/
│   │       └── 001_initial.sql  # [BUILT] schema for market_data, fundamentals, trades, signals
│   ├── analysis/                # financial analysis engine (not yet built)
│   │   ├── dcf_model.py         # discounted cash flow with Monte Carlo simulation
│   │   ├── ratio_analysis.py    # P/E, P/S, PEG, debt-to-equity, FCF margin
│   │   ├── sensitivity.py       # sensitivity matrix (growth rate x terminal multiple)
│   │   ├── earnings_signals.py  # earnings surprise, guidance, estimate revisions
│   │   ├── insider_activity.py  # SEC Form 4 — insider buys/sells
│   │   └── ai_summary.py       # Groq free tier — natural language analysis summary
│   ├── indicators/              # technical indicator library (not yet built)
│   │   ├── trend.py             # SMA, EMA, MACD, ADX, Supertrend
│   │   ├── momentum.py          # RSI, Stochastic, CCI, Williams %R, ROC
│   │   ├── volatility.py        # Bollinger Bands, ATR, Keltner Channel
│   │   ├── volume.py            # OBV, VWAP, Volume Profile, MFI
│   │   ├── structure.py         # Fair Value Gaps, Order Blocks, Structure Breaks
│   │   ├── support_resistance.py # Pivot points, Fibonacci levels, S/R zones
│   │   └── crypto_specific.py   # Funding rate, open interest, exchange flows, NVT
│   ├── quant/                   # quantitative engines (not yet built)
│   │   ├── monte_carlo.py       # Monte Carlo simulation + variance reduction
│   │   ├── importance_sampling.py # rare event estimation for tail risk
│   │   ├── particle_filter.py   # Sequential Monte Carlo for real-time updating
│   │   ├── copula_models.py     # Gaussian, t-copula, Clayton for correlation
│   │   ├── risk_metrics.py      # VaR, Expected Shortfall, max drawdown
│   │   └── brier_score.py       # calibration tracking for predictions
│   ├── strategies/              # strategy framework (not yet built)
│   │   ├── base_strategy.py     # abstract StrategyInterface
│   │   ├── strategy_loader.py   # loads strategy configs from JSON files
│   │   ├── strategy_pool.py     # manages multiple concurrent strategies
│   │   └── backtest.py          # backtesting engine — scores strategies
│   ├── agents/                  # runtime agentic team (not yet built)
│   │   ├── orchestrator.py      # main loop — coordinates all agents
│   │   ├── analyst_agent.py     # runs financial analysis, scores stocks
│   │   ├── strategy_agent.py    # generates trade signals from indicators + analysis
│   │   ├── risk_agent.py        # approves/rejects trades, checks portfolio risk
│   │   ├── executor_agent.py    # places orders through broker layer
│   │   └── tools.py             # MCP-style tools each agent can call
│   ├── evolution/               # autoresearch loop (not yet built)
│   │   ├── evolution_engine.py  # reads logs -> proposes mutations -> backtests -> keeps/discards
│   │   ├── strategy_mutator.py  # LLM-powered strategy config mutation
│   │   ├── report_generator.py  # weekly evolution reports (markdown)
│   │   └── documentation.py     # auto-updates docs after each evolution cycle
│   └── dashboard/               # web dashboard (not yet built)
│       ├── app.py               # FastAPI backend serving analysis + trading data
│       └── templates/           # HTML templates or React frontend
├── tests/                       # test suite — 173 tests passing
│   ├── __init__.py
│   ├── test_db.py               # [BUILT] 12 tests — pool init, migrations, masking
│   ├── test_symbols.py          # [BUILT] 24 tests — symbol conversion, universes
│   ├── test_resilience.py       # [BUILT] 9 tests — retry, circuit breaker
│   ├── test_validators.py       # [BUILT] 28 tests — bounds, ohlcv, fundamentals, sentiment
│   ├── test_market_data.py      # [BUILT] 21 tests — alpaca, yfinance, backfill, store
│   ├── test_fundamentals.py     # [BUILT] 19 tests — fetch, store, sector averages
│   ├── test_sec_filings.py      # [BUILT] 17 tests — insider trades, data quality
│   ├── test_regime_detector.py  # [BUILT] 18 tests — classify, indicators, backfill
│   └── conftest.py              # [BUILT] shared fixtures
├── reports/                     # auto-generated evolution reports
│   └── .gitkeep
└── docs/
    └── (empty — to be populated)
```

## gstack

Use /browse from gstack for all web browsing. Never use mcp__claude-in-chrome__* tools.
Available skills: /office-hours, /plan-ceo-review, /plan-eng-review, /plan-design-review,
/design-consultation, /review, /ship, /land-and-deploy, /qa, /qa-only,
/design-review, /retro, /investigate, /document-release, /codex, /cso,
/autoplan, /careful, /freeze, /guard, /unfreeze, /gstack-upgrade.

## Critical Design Decisions

### 1. Modular Broker Interface

Every broker implements this exact interface. Strategies NEVER touch brokers directly.

```python
class BrokerInterface(ABC):
    @abstractmethod
    async def get_price(self, symbol: str) -> float: ...
    @abstractmethod
    async def place_order(self, symbol: str, qty: float, side: str, order_type: str = "market") -> dict: ...
    @abstractmethod
    async def get_positions(self) -> list[dict]: ...
    @abstractmethod
    async def get_account_balance(self) -> dict: ...
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...
    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict: ...
    @abstractmethod
    async def stream_prices(self, symbols: list[str], callback) -> None: ...
```

BrokerFactory routes symbols: stocks (AAPL, MSFT) → AlpacaBroker. Crypto (BTC-USD, ETH-USD) → AlpacaBroker (initially), can be switched to CoinbaseBroker later. PaperBroker wraps any real broker for simulated trading.

### 2. Strategy Config Format

Each strategy is a JSON file in configs/strategies/. The LLM evolution agent reads, modifies, and creates these — it never touches Python code directly.

```json
{
  "id": "strategy_001",
  "name": "Bollinger_PE_Oversold",
  "version": 3,
  "created_by": "evolution_agent",
  "parent_id": "strategy_base_001",
  "description": "Buy when Bollinger bands show oversold AND P/E is below sector average",
  "asset_class": "stocks",
  "universe": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA"],

  "fundamental_filters": {
    "pe_ratio_max": 35,
    "pe_vs_sector": "below_average",
    "revenue_growth_min": 0.02,
    "fcf_margin_min": 0.15,
    "debt_to_equity_max": 1.5,
    "insider_buying_recent": true,
    "dcf_upside_min": 0.10
  },

  "entry_conditions": {
    "operator": "AND",
    "signals": [
      {"indicator": "bollinger_bands", "condition": "price_below_lower", "lookback": 20, "std_dev": 2.0},
      {"indicator": "rsi", "condition": "below", "threshold": 35, "period": 14},
      {"indicator": "volume", "condition": "above_average", "multiplier": 1.3, "period": 20}
    ]
  },

  "exit_conditions": {
    "take_profit": {"indicator": "bollinger_bands", "condition": "price_above_middle"},
    "stop_loss": {"type": "atr_trailing", "multiplier": 2.0, "period": 14},
    "time_exit": {"max_holding_days": 30}
  },

  "position_sizing": {
    "method": "kelly_fraction",
    "max_position_pct": 0.08,
    "kelly_fraction": 0.25
  },

  "metadata": {
    "generation": 3,
    "backtest_sharpe": 1.42,
    "backtest_max_drawdown": -0.12,
    "backtest_win_rate": 0.58,
    "brier_score": 0.18,
    "paper_trade_days": 14,
    "paper_trade_pnl": 0.034,
    "status": "paper_trading"
  }
}
```

### 3. Fundamental Base Layer (Always Present)

Every strategy, regardless of its technical signals, receives these fundamental inputs. The analysis engine computes these for all stocks in the universe daily:

- P/E ratio (trailing + forward) vs sector average
- P/S ratio vs sector average
- PEG ratio
- Revenue growth rate (1Y, 3Y, 5Y)
- Free cash flow margin
- Debt-to-equity ratio
- DCF fair value estimate (Monte Carlo distribution — not a single number)
- DCF upside/downside percentage
- Earnings surprise history (last 4 quarters)
- Insider buying/selling (SEC Form 4, last 90 days)
- Analyst consensus target vs current price
- Sensitivity matrix values at different growth/terminal assumptions

For crypto, the fundamental layer uses (via CoinGecko, Dune Analytics, DefiLlama):
- NVT ratio (network value to transactions) — Dune
- Active addresses trend — Dune
- Exchange inflow/outflow ratio — Dune
- Funding rates (perp markets) — CoinGecko
- Hash rate trend (for BTC) — CoinGecko
- Stablecoin supply ratio — DefiLlama
- TVL trends by protocol — DefiLlama
- DEX volume — DefiLlama

### 4. Agent Communication Pattern

Agents communicate through a shared PostgreSQL database (Neon free tier) with these tables:

- `analysis_scores` — analyst agent writes, strategy agent reads
- `trade_signals` — strategy agent writes, risk agent reads
- `approved_trades` — risk agent writes, executor agent reads
- `trade_log` — executor agent writes, all agents + evolution engine reads
- `strategy_scores` — backtester writes, evolution engine reads
- `evolution_log` — evolution engine writes, reports use

Each agent runs on a schedule:
- Analyst agent: every 1 hour during market hours, every 4 hours off-hours
- Strategy agent: every 15 minutes during market hours, every 1 hour for crypto
- Risk agent: triggered on each trade signal (event-driven)
- Executor agent: triggered on each approved trade (event-driven)
- Evolution engine: nightly at midnight ET via cron

### 5. The Autoresearch Loop

The evolution engine follows Karpathy's autoresearch pattern exactly:

1. READ: Load all strategy scores from the last scoring period
2. RANK: Sort strategies by composite score (Sharpe * 0.4 + win_rate * 0.3 - max_drawdown * 0.3)
3. KILL: Remove bottom 25% of strategies from the pool
4. MUTATE: For each killed strategy, use Claude Haiku to propose a mutation:
   - Read the killed strategy's config
   - Read the top-performing strategy's config
   - Read the trade log showing what went wrong
   - Propose a new config that combines the best elements
5. BACKTEST: Run each new config against historical data
6. SCORE: If new config scores higher than median, add to pool as "paper_trading"
7. PROMOTE: If a paper-trading strategy outperforms for 2+ weeks, promote to "live"
8. DOCUMENT: Write markdown report to reports/ folder

The `evolution.md` file (equivalent to autoresearch's program.md) tells the LLM:
- What metrics to optimize (Sharpe ratio primary, max drawdown constraint)
- What constraints to respect (max 10% per position, dual-track: fundamental-gated + momentum-only)
- What experiments to try (combine indicators from winners, try new timeframes)
- What NOT to do (never exceed risk limits per track)

### 6. Quant Engine Integration

Monte Carlo and other quant techniques are NOT separate modules — they're integrated:

- **DCF + Monte Carlo**: Instead of one fair value, dcf_model.py runs 10,000 simulations with randomized growth rates and margins. Output: probability distribution of fair value. This feeds directly into fundamental_filters.dcf_upside_min.

- **Copula models**: When the risk agent checks a proposed trade, it computes tail dependence between the new position and existing portfolio using a Student-t copula. If adding MSFT to a portfolio already holding AAPL and GOOG would create >0.3 tail dependence, the trade is rejected or position sized down.

- **Importance sampling**: Used in the backtester to estimate tail risk (probability of >20% drawdown) without needing millions of samples.

- **Particle filter**: Used by the strategy agent when processing real-time data. When earnings drop, the filter smoothly updates the probability estimate instead of jerking to the new price.

- **Brier score**: Tracked for every strategy's predictions in the trade log. The evolution engine uses this as one of its scoring metrics.

### 7. Technical Indicator Implementation Notes

All indicators use pandas/numpy. TA-Lib is optional (faster but harder to install). The structure.py module implements ICT/smart money concepts:

- **Fair Value Gap (FVG)**: Three-candle pattern where candle 1's high < candle 3's low (bullish) or candle 1's low > candle 3's high (bearish). Returns list of unfilled gaps with price range and age.

- **Order Block**: Last bullish candle before a bearish impulse move (or vice versa). Detected by finding the candle before a significant directional move (>2 ATR) in the opposite direction. Returns price zone (high/low of that candle).

- **Structure Break (BOS/CHoCH)**: Track swing highs and swing lows. Break of Structure = price breaking a swing point in the trend direction. Change of Character = price breaking a swing point against the trend direction. Uses zigzag with ATR-based threshold.

These work on any timeframe and any asset (stocks, crypto, ETFs).

## Environment Variables

```
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper trading first!
ANTHROPIC_API_KEY=xxx          # for Claude Haiku (evolution agent)
GROQ_API_KEY=xxx               # free tier (analyst agent summaries)
SEC_EDGAR_USER_AGENT=YourName your@email.com  # required by SEC
FINNHUB_API_KEY=xxx            # free tier (events + sentiment)
DUNE_API_KEY=xxx               # free tier (on-chain analytics)
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler.neon.tech/neondb?sslmode=require
LOG_LEVEL=INFO
EVOLUTION_SCHEDULE=0 0 * * *   # midnight ET daily
MAX_POSITION_PCT=0.08          # 8% max per position (fundamental-gated track)
MAX_PORTFOLIO_RISK=0.25        # 25% max total risk
TELEGRAM_BOT_TOKEN=xxx         # optional — pipeline failure alerts
TELEGRAM_CHAT_ID=xxx           # optional
```

## Build Order

Build modules in this order. Each module should be testable independently before moving on.

1. **Data layer** (src/data/) — db.py, symbols.py, resilience.py, validators.py, market_data.py, fundamentals.py, sec_filings.py, regime_detector.py
2. **Analysis engine** (src/analysis/) — ratio_analysis.py, dcf_model.py, sensitivity.py
3. **Indicator library** (src/indicators/) — start with trend.py, momentum.py, volatility.py
4. **Broker layer** (src/brokers/) — base.py, paper_broker.py, alpaca_broker.py
5. **Strategy framework** (src/strategies/) — base_strategy.py, strategy_loader.py, backtest.py
6. **Quant engine** (src/quant/) — monte_carlo.py, risk_metrics.py, brier_score.py
7. **Agent orchestrator** (src/agents/) — orchestrator.py and all agents
8. **Evolution engine** (src/evolution/) — evolution_engine.py, strategy_mutator.py
9. **Advanced indicators** (src/indicators/structure.py) — FVG, order blocks, BOS/CHoCH
10. **Advanced quant** (src/quant/) — copula_models.py, particle_filter.py, importance_sampling.py
11. **Dashboard** (src/dashboard/) — FastAPI app with analysis + trading views

## Testing

Run: `python -m pytest tests/ -v`

173 tests passing across 9 test files. Every module has tests. Key patterns:
- **asyncpg pool mocking**: use `_mock_pool()` helper — `MagicMock` for pool, `AsyncMock` for context manager and connection
- **external API mocking**: patch httpx.AsyncClient for alpaca, patch yfinance.Ticker for yfinance, patch edgartools for SEC
- **data validation**: test both valid and invalid inputs, verify graceful handling of edge cases
- Broker tests: use PaperBroker to verify order flow without real API calls
- Strategy tests: use known historical data with known outcomes
- Analysis tests: verify DCF math against manually calculated values
- Indicator tests: verify against known indicator values from TradingView
- Backtest tests: verify P&L calculation, drawdown calculation, Sharpe calculation
- Evolution tests: verify mutation produces valid strategy configs

## Dependencies

```
# Core
alpaca-py>=0.30.0          # Alpaca broker API
yfinance>=0.2.0            # Yahoo Finance data
edgartools>=2.0.0           # SEC EDGAR filings (free, no API key)
finnhub-python>=2.4.0      # Finnhub events + sentiment (free tier)
dune-client>=1.0.0         # Dune Analytics on-chain queries (free tier)
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0              # Stats, copulas, distributions
asyncpg>=0.29.0            # Async PostgreSQL (Neon)
httpx>=0.27.0              # Async HTTP client
python-dotenv>=1.0.0
structlog>=24.1.0          # Structured JSON logging
exchange_calendars>=4.5.0  # NYSE market calendar

# LLM
anthropic>=0.40.0          # Claude API (Haiku for evolution)
groq>=0.11.0               # Groq free tier (analyst summaries)

# Technical analysis
pandas-ta>=0.3.14          # Technical indicators (pure Python, no TA-Lib needed)

# Dashboard (phase 2)
fastapi>=0.100.0
uvicorn>=0.22.0
jinja2>=3.1.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

## Style Guide

- Python 3.11+, type hints everywhere
- Async/await for all IO operations (broker calls, data fetches, DB queries)
- Dataclasses for data objects, Pydantic for config validation
- Logging with structured output (JSON format for machine parsing)
- Every function has a docstring explaining what it does, its inputs, and outputs
- No hardcoded values — everything configurable via .env or config files
