# CLAUDE.md — Quant Trading System Blueprint

## Project Overview

Self-improving agentic trading system for US stocks (via Alpaca, commission-free) and crypto (via Alpaca initially, Coinbase later). The system runs 24/7 on a VPS (~$8/month), uses free data APIs, and evolves its own strategies using a Karpathy autoresearch-inspired feedback loop.

**Core principle**: Every strategy gets a fundamental base layer (always present) + a technical signal layer (varies per strategy). Fundamentals are the cornerstone. Technicals are the timing mechanism.

**Budget constraint**: Total infrastructure cost must stay under $25/month. Use free tiers aggressively (Groq free for lightweight LLM, edgartools for SEC data, yfinance for fundamentals, Alpaca free tier for market data, Finnhub for events/sentiment, CoinGecko for crypto market data, DefiLlama for DeFi flows, Dune Analytics for on-chain).

## Progress

- **Phase 1** (foundation): COMPLETE — db, symbols, resilience, validators, migrations
- **Phase 2** (data layer + regime detection): COMPLETE — market_data, fundamentals, sec_filings, regime_detector, backfill script
- **Phase 3** (analysis engine): COMPLETE — ratio_analysis, dcf_model, sensitivity, earnings_signals, insider_activity, ai_summary
- **Phase 4** (indicators + broker layer): COMPLETE — trend, momentum, volatility, volume indicators + paper_broker, alpaca_broker, broker_factory
- **366 tests passing** (98 Phase 1 + 75 Phase 2 + 96 Phase 3 + 97 Phase 4)

## Phases

### Phase 1: Foundation [COMPLETE]
db.py, symbols.py, resilience.py, validators.py, migrations/001_initial.sql

### Phase 2: Data Layer + Regime Detection [COMPLETE]
market_data.py, fundamentals.py, sec_filings.py, regime_detector.py, scripts/backfill.py

### Phase 3: Analysis Engine [COMPLETE]
ratio_analysis.py, dcf_model.py, sensitivity.py, earnings_signals.py, insider_activity.py, ai_summary.py

### Phase 4: Indicators + Broker Layer [COMPLETE]
**Technical indicators** (src/indicators/):
- trend.py — SMA, EMA, MACD, ADX, Supertrend
- momentum.py — RSI, Stochastic, CCI, Williams %R, ROC
- volatility.py — Bollinger Bands, ATR, Keltner Channel
- volume.py — OBV, VWAP, Volume Profile, MFI

**Broker layer** (src/brokers/):
- base.py — abstract BrokerInterface
- paper_broker.py — simulated broker for backtesting
- alpaca_broker.py — Alpaca REST/WebSocket (stocks + crypto)
- broker_factory.py — routes symbols to correct broker

### Phase 5: Strategy Framework + Backtesting
**Strategy framework** (src/strategies/):
- base_strategy.py — abstract StrategyInterface
- strategy_loader.py — loads strategy configs from JSON files
- strategy_pool.py — manages multiple concurrent strategies
- backtest.py — backtesting engine with P&L, drawdown, Sharpe calculation

### Phase 6: Quant Engine
**Core Monte Carlo** (src/quant/):
- monte_carlo.py — production MC with variance reduction (antithetic variates, control variates, stratified sampling). Antithetic: use Z and -Z pairs for free ~50-75% variance reduction. Control variate: use Black-Scholes closed-form as baseline correction. Stratified: partition via quantiles, sample within each stratum, combine with Neyman allocation. Stack all three for 100-500x reduction over crude MC.
- importance_sampling.py — exponential tilting for tail risk estimation. Shifts the sampling distribution toward the rare region (e.g., >20% crash), then corrects with likelihood ratio. Solves the problem where crude MC gives 0/1 hits on extreme events. 100 IS samples can beat 1M crude samples.
- risk_metrics.py — VaR (parametric + historical + MC), Expected Shortfall (CVaR), max drawdown, Sharpe, Sortino. EVT-based tail estimation using Generalized Pareto for extreme quantiles.
- brier_score.py — calibration tracking for all strategy predictions. Brier < 0.20 = good, < 0.10 = excellent. Tracked per-strategy in trade_log, used by evolution engine as scoring metric.

**Real-time + correlation** (src/quant/):
- particle_filter.py — Sequential Monte Carlo (bootstrap filter) for real-time probability updating. Maintains N particles as hypotheses about true state, reweights on each observation via likelihood, systematic resampling when ESS < N/2. Operates in logit space to keep probabilities bounded. Used by strategy agent to smooth noisy signals (earnings drops, price spikes) instead of jerking to raw values.
- copula_models.py — Gaussian copula (no tail dependence — baseline only), Student-t copula (symmetric tail dependence, λ > 0 for finite ν), Clayton copula (lower tail dependence — crash correlation). Risk agent uses t-copula to check tail dependence between proposed trade and existing portfolio. If tail dependence > 0.3, reject or size down. Gaussian copula underestimates extreme co-movement by 2-5x — this is what blew up in 2008.

### Phase 7: Agents + Evolution
**Agent orchestrator** (src/agents/):
- orchestrator.py — main loop coordinating all agents on schedule
- analyst_agent.py — runs analysis engine, writes to analysis_scores
- strategy_agent.py — generates trade signals from indicators + analysis + particle filter
- risk_agent.py — approves/rejects trades via copula-based portfolio risk check
- executor_agent.py — places orders through broker layer
- tools.py — MCP-style tools each agent can call

**Evolution engine** (src/evolution/):
- evolution_engine.py — Karpathy autoresearch loop (read → rank → kill → mutate → backtest → score → promote → document)
- strategy_mutator.py — LLM-powered strategy config mutation via Claude Haiku
- report_generator.py — weekly evolution reports (markdown)
- documentation.py — auto-updates docs after each evolution cycle

### Phase 8: Advanced Indicators + Data Sources
**Advanced indicators** (src/indicators/):
- structure.py — ICT/smart money: Fair Value Gaps, Order Blocks, BOS/CHoCH
- support_resistance.py — Pivot points, Fibonacci levels, S/R zones
- crypto_specific.py — Funding rate, open interest, exchange flows, NVT

**Additional data sources** (src/data/):
- news_sentiment.py — Finnhub events + NLP sentiment scoring
- crypto_data.py — CoinGecko price/market data + DefiLlama TVL/flows
- crypto_onchain.py — Dune Analytics custom on-chain queries

### Phase 9: Dashboard
**Web dashboard** (src/dashboard/):
- app.py — FastAPI backend serving analysis + trading data
- templates/ — HTML templates or React frontend
- Real-time portfolio view, analysis scores, trade log, evolution reports

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
│   │   ├── news_sentiment.py    # [Phase 8] Finnhub events + sentiment layer
│   │   ├── crypto_data.py       # [Phase 8] CoinGecko price/market data + DefiLlama flows
│   │   ├── crypto_onchain.py    # [Phase 8] Dune Analytics custom on-chain queries
│   │   └── migrations/
│   │       └── 001_initial.sql  # [BUILT] schema for market_data, fundamentals, trades, signals
│   ├── analysis/                # financial analysis engine
│   │   ├── ratio_analysis.py    # [BUILT] P/E, P/S, PEG, debt-to-equity, FCF margin scoring (0-100)
│   │   ├── dcf_model.py         # [BUILT] DCF with Monte Carlo simulation (10k runs, p10/median/p90)
│   │   ├── sensitivity.py       # [BUILT] growth rate x terminal multiple sensitivity grid
│   │   ├── earnings_signals.py  # [BUILT] earnings surprise detection, beat streaks, signal scoring
│   │   ├── insider_activity.py  # [BUILT] insider buy/sell aggregation, cluster detection, title weighting
│   │   └── ai_summary.py       # [BUILT] Groq free tier summary with structured fallback
│   ├── indicators/              # technical indicator library
│   │   ├── __init__.py
│   │   ├── trend.py             # [BUILT] SMA, EMA, MACD, ADX, Supertrend
│   │   ├── momentum.py          # [BUILT] RSI, Stochastic, CCI, Williams %R, ROC
│   │   ├── volatility.py        # [BUILT] Bollinger Bands, ATR, Keltner Channel
│   │   ├── volume.py            # [BUILT] OBV, VWAP, Volume Profile, MFI
│   │   ├── structure.py         # [Phase 8] Fair Value Gaps, Order Blocks, Structure Breaks
│   │   ├── support_resistance.py # [Phase 8] Pivot points, Fibonacci levels, S/R zones
│   │   └── crypto_specific.py   # [Phase 8] Funding rate, open interest, exchange flows, NVT
│   ├── brokers/                 # modular broker layer — swappable
│   │   ├── __init__.py
│   │   ├── base.py              # [BUILT] abstract BrokerInterface + Order/Position/AccountBalance dataclasses
│   │   ├── alpaca_broker.py     # [BUILT] Alpaca REST implementation (stocks + crypto)
│   │   ├── coinbase_broker.py   # Coinbase implementation (future, crypto)
│   │   ├── paper_broker.py      # [BUILT] simulated broker for backtesting
│   │   └── broker_factory.py    # [BUILT] routes symbols to correct broker
│   ├── strategies/              # [Phase 5] strategy framework
│   │   ├── base_strategy.py     # abstract StrategyInterface
│   │   ├── strategy_loader.py   # loads strategy configs from JSON files
│   │   ├── strategy_pool.py     # manages multiple concurrent strategies
│   │   └── backtest.py          # backtesting engine — scores strategies
│   ├── quant/                   # [Phase 6] quantitative engines
│   │   ├── monte_carlo.py       # MC simulation + variance reduction (antithetic + control + stratified)
│   │   ├── importance_sampling.py # exponential tilting for tail risk (100x-10000x variance reduction)
│   │   ├── particle_filter.py   # Sequential Monte Carlo for real-time updating (bootstrap filter)
│   │   ├── copula_models.py     # Gaussian, Student-t, Clayton copulas for tail dependence
│   │   ├── risk_metrics.py      # VaR, CVaR, max drawdown, EVT-based tail estimation
│   │   └── brier_score.py       # calibration tracking for strategy predictions
│   ├── agents/                  # [Phase 7] runtime agentic team
│   │   ├── orchestrator.py      # main loop — coordinates all agents
│   │   ├── analyst_agent.py     # runs financial analysis, scores stocks
│   │   ├── strategy_agent.py    # generates trade signals from indicators + analysis
│   │   ├── risk_agent.py        # approves/rejects trades, checks portfolio risk
│   │   ├── executor_agent.py    # places orders through broker layer
│   │   └── tools.py             # MCP-style tools each agent can call
│   ├── evolution/               # [Phase 7] autoresearch loop
│   │   ├── evolution_engine.py  # reads logs -> proposes mutations -> backtests -> keeps/discards
│   │   ├── strategy_mutator.py  # LLM-powered strategy config mutation
│   │   ├── report_generator.py  # weekly evolution reports (markdown)
│   │   └── documentation.py     # auto-updates docs after each evolution cycle
│   └── dashboard/               # [Phase 9] web dashboard
│       ├── app.py               # FastAPI backend serving analysis + trading data
│       └── templates/           # HTML templates or React frontend
├── tests/                       # test suite — 366 tests passing
│   ├── __init__.py
│   ├── test_db.py               # [BUILT] 12 tests — pool init, migrations, masking
│   ├── test_symbols.py          # [BUILT] 24 tests — symbol conversion, universes
│   ├── test_resilience.py       # [BUILT] 9 tests — retry, circuit breaker
│   ├── test_validators.py       # [BUILT] 28 tests — bounds, ohlcv, fundamentals, sentiment
│   ├── test_market_data.py      # [BUILT] 21 tests — alpaca, yfinance, backfill, store
│   ├── test_fundamentals.py     # [BUILT] 19 tests — fetch, store, sector averages
│   ├── test_sec_filings.py      # [BUILT] 17 tests — insider trades, data quality
│   ├── test_regime_detector.py  # [BUILT] 18 tests — classify, indicators, backfill
│   ├── test_ratio_analysis.py   # [BUILT] 27 tests — pe/ps/peg/fcf/de scoring, composite, db
│   ├── test_dcf_model.py        # [BUILT] 14 tests — simulation, assumptions, storage
│   ├── test_sensitivity.py      # [BUILT] 7 tests — matrix shape, monotonicity, driver detection
│   ├── test_earnings_signals.py # [BUILT] 13 tests — surprise, streaks, fetch, pipeline
│   ├── test_insider_activity.py # [BUILT] 17 tests — weighting, clusters, signals, db
│   ├── test_ai_summary.py       # [BUILT] 10 tests — prompt, fallback, groq mock, errors
│   ├── test_trend.py            # [BUILT] 14 tests — sma, ema, macd, adx, supertrend
│   ├── test_momentum.py         # [BUILT] 13 tests — rsi, stochastic, cci, williams_r, roc
│   ├── test_volatility.py       # [BUILT] 11 tests — bollinger, atr, keltner
│   ├── test_volume.py           # [BUILT] 13 tests — obv, vwap, volume_profile, mfi
│   ├── test_paper_broker.py     # [BUILT] 19 tests — orders, positions, balance, cancel, stream
│   ├── test_alpaca_broker.py    # [BUILT] 7 tests — mocked rest api, parse_order
│   ├── test_broker_factory.py   # [BUILT] 7 tests — paper/live mode, routing
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

### 6. Quant Engine — Mathematical Foundations

The quant engine implements production-grade numerical methods. These are not academic exercises — they directly drive trading decisions.

**Monte Carlo + Variance Reduction (monte_carlo.py)**
The DCF model already uses basic MC. The quant engine upgrades this with three stacking techniques:
- Antithetic variates: for every Z, also evaluate -Z. Free 50-75% variance reduction on monotone payoffs. Zero extra cost.
- Control variates: if a closed-form approximation exists (e.g. Black-Scholes for options), use it as a baseline: θ_CV = θ_MC - c*(f_MC - f_exact). Corrects systematic bias.
- Stratified sampling: partition the probability space into J strata via quantiles, sample within each, combine with weights. Neyman allocation (n_j ∝ ω_j * σ_j) oversamples high-variance strata.
Stacking all three: 100-500x variance reduction over crude MC. This is table stakes for production, not optional.

**Importance Sampling (importance_sampling.py)**
For tail risk contracts (P(crash > 20%) ≈ 0.003), crude MC gives 0 or 1 hits in 100k samples — useless.
Exponential tilting replaces the original measure with one centered on the rare event, then corrects via likelihood ratio. Choose tilt parameter γ so the rare threshold becomes ~1 std dev away.
Result: 100 IS samples can beat 1,000,000 crude samples. Variance reduction of 100-10,000x on extreme events.

**Particle Filter (particle_filter.py)**
Bootstrap filter for real-time Bayesian updating. State evolves via logit random walk (bounded probabilities), observations are noisy readings.
Algorithm: propagate particles → reweight by likelihood → normalize → resample if ESS < N/2.
Systematic resampling (lower variance than multinomial). Smooths noisy signals — when price spikes on a single trade, the filter tempers the update.

**Copula Models (copula_models.py)**
Sklar's theorem: any joint distribution F(x₁,...,xₙ) = C(F₁(x₁),...,Fₙ(xₙ)) where C is the copula.
- Gaussian copula: tail dependence λ = 0. Catastrophically wrong for correlated assets in crisis. Baseline only.
- Student-t copula: symmetric tail dependence. With ν=4, ρ=0.6 → λ ≈ 0.18 (18% probability of extreme co-movement). This is 2-5x higher than Gaussian predicts.
- Clayton copula: lower tail dependence only (crash correlation). λ_L = 2^(-1/θ).
Risk agent uses t-copula to gate trades. If adding a position creates >0.3 tail dependence with existing portfolio, reject or size down.

**Risk Metrics (risk_metrics.py)**
- VaR: parametric (assumes distribution), historical (empirical quantile), Monte Carlo (simulated)
- Expected Shortfall (CVaR): average loss beyond VaR — better for fat tails
- EVT: fit Generalized Pareto to tail exceedances for extreme quantile estimation
- Max drawdown, Sharpe, Sortino, Calmar

**Brier Score (brier_score.py)**
BS = mean((predicted - actual)²). Measures calibration quality.
Below 0.20 = good. Below 0.10 = excellent. Best election forecasters achieve 0.06-0.12.
Tracked per-strategy, used by evolution engine to rank and kill poorly calibrated strategies.

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

## Testing

Run: `python -m pytest tests/ -v`

366 tests passing across 22 test files. Every module has tests. Key patterns:
- **asyncpg pool mocking**: use `_mock_pool()` helper — `MagicMock` for pool, `AsyncMock` for context manager and connection
- **external API mocking**: patch httpx.AsyncClient for alpaca, patch yfinance.Ticker for yfinance, patch edgartools for SEC
- **data validation**: test both valid and invalid inputs, verify graceful handling of edge cases
- **numpy seeded RNG**: use `np.random.default_rng(42)` for deterministic MC tests
- Analysis tests: verify DCF math, scoring monotonicity, sensitivity grid shape
- Broker tests: use PaperBroker to verify order flow without real API calls
- Strategy tests: use known historical data with known outcomes
- Indicator tests: verify against known indicator values from TradingView
- Backtest tests: verify P&L calculation, drawdown calculation, Sharpe calculation
- Quant tests: verify variance reduction ratios, copula tail dependence, particle filter convergence
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

# Dashboard (phase 9)
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
- Comments: lowercase, concise, `/` prefix
- Commits: 1-2 lines lowercase, no prefixes (no feat:, fix:, docs:, chore:), just say what changed
