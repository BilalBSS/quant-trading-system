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
- **Phase 5** (strategy + backtesting): COMPLETE — base_strategy, strategy_loader, strategy_pool, backtest + 10 seed configs
- **Phase 6** (quant engine): COMPLETE — monte_carlo, importance_sampling, risk_metrics, brier_score, particle_filter, copula_models
- **Phase 7** (agents + evolution): COMPLETE — orchestrator, analyst_agent, strategy_agent, risk_agent, executor_agent, tools + evolution_engine, strategy_mutator, report_generator, documentation
- **1273 tests passing** (129 Phase 1 + 125 Phase 2 + 152 Phase 3 + 164 Phase 4 + 329 Phase 5 + 168 Phase 6 + 206 Phase 7)

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

### Phase 5: Strategy Framework + Backtesting [COMPLETE]
**Strategy framework** (src/strategies/):
- base_strategy.py — abstract StrategyInterface + ConfigDrivenStrategy (evaluates JSON configs against indicators + analysis)
- strategy_loader.py — Pydantic-validated JSON config loader with track constraints (fundamental-gated vs momentum-only)
- strategy_pool.py — manages N concurrent strategies with ranking, bottom quartile detection, lifecycle tracking
- backtest.py — backtesting engine: anti-lookahead (signal at close, fill at next open), Sharpe/Sortino/Calmar/drawdown/win rate/profit factor

### Phase 6: Quant Engine [COMPLETE]
**Core Monte Carlo** (src/quant/):
- monte_carlo.py — production MC with variance reduction (antithetic variates, control variates, stratified sampling). Antithetic: use Z and -Z pairs for free ~50-75% variance reduction. Control variate: use Black-Scholes closed-form as baseline correction. Stratified: partition via quantiles, sample within each stratum, combine with Neyman allocation. Stack all three for 100-500x reduction over crude MC.
- importance_sampling.py — exponential tilting for tail risk estimation. Shifts the sampling distribution toward the rare region (e.g., >20% crash), then corrects with likelihood ratio. Solves the problem where crude MC gives 0/1 hits on extreme events. 100 IS samples can beat 1M crude samples.
- risk_metrics.py — VaR (parametric + historical + MC), Expected Shortfall (CVaR), max drawdown, Sharpe, Sortino. EVT-based tail estimation using Generalized Pareto for extreme quantiles.
- brier_score.py — calibration tracking for all strategy predictions. Brier < 0.20 = good, < 0.10 = excellent. Tracked per-strategy in trade_log, used by evolution engine as scoring metric.

**Real-time + correlation** (src/quant/):
- particle_filter.py — Sequential Monte Carlo (bootstrap filter) for real-time probability updating. Maintains N particles as hypotheses about true state, reweights on each observation via likelihood, systematic resampling when ESS < N/2. Operates in logit space to keep probabilities bounded. Used by strategy agent to smooth noisy signals (earnings drops, price spikes) instead of jerking to raw values.
- copula_models.py — Gaussian copula (no tail dependence — baseline only), Student-t copula (symmetric tail dependence, λ > 0 for finite ν), Clayton copula (lower tail dependence — crash correlation). Risk agent uses t-copula to check tail dependence between proposed trade and existing portfolio. If tail dependence > 0.3, reject or size down. Gaussian copula underestimates extreme co-movement by 2-5x — this is what blew up in 2008.

### Phase 7: Agents + Evolution [COMPLETE]
**Agent orchestrator** (src/agents/):
- orchestrator.py — main loop coordinating all agents on schedule (NYSE market hours via exchange_calendars)
- analyst_agent.py — runs analysis engine (ratio, dcf, earnings, insider, ai_summary), writes to analysis_scores
- strategy_agent.py — generates trade signals from indicators + analysis + particle filter signal smoothing
- risk_agent.py — approves/rejects trades via position sizing, portfolio risk limits, copula-based tail dependence check
- executor_agent.py — places orders through broker layer, logs to trade_log
- tools.py — shared async DB helpers for all agents (store/fetch for all pipeline tables)

**Evolution engine** (src/evolution/):
- evolution_engine.py — Karpathy autoresearch loop (read → rank → kill → mutate → backtest → score → promote → document), parallel backtesting via asyncio.gather
- strategy_mutator.py — Claude Haiku strategy mutation with 3-retry loop, random tweak fallback
- report_generator.py — markdown evolution reports per generation
- documentation.py — auto-updates CHANGELOG after each evolution cycle

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
│   ├── strategies/              # strategy framework
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # [BUILT] abstract StrategyInterface + ConfigDrivenStrategy
│   │   ├── strategy_loader.py   # [BUILT] Pydantic-validated JSON config loader
│   │   ├── strategy_pool.py     # [BUILT] manages N concurrent strategies with ranking
│   │   └── backtest.py          # [BUILT] backtesting engine — anti-lookahead, full metrics
│   ├── quant/                   # quantitative engines
│   │   ├── monte_carlo.py       # [BUILT] MC simulation + variance reduction (antithetic + control + stratified)
│   │   ├── importance_sampling.py # [BUILT] exponential tilting for tail risk (100x-10000x variance reduction)
│   │   ├── particle_filter.py   # [BUILT] Sequential Monte Carlo for real-time updating (bootstrap filter)
│   │   ├── copula_models.py     # [BUILT] Gaussian, Student-t, Clayton copulas for tail dependence
│   │   ├── risk_metrics.py      # [BUILT] VaR, CVaR, max drawdown, EVT-based tail estimation
│   │   └── brier_score.py       # [BUILT] calibration tracking for strategy predictions
│   ├── agents/                  # runtime agentic team
│   │   ├── orchestrator.py      # [BUILT] main loop — coordinates all agents on NYSE schedule
│   │   ├── analyst_agent.py     # [BUILT] runs financial analysis pipeline, scores stocks
│   │   ├── strategy_agent.py    # [BUILT] generates trade signals with particle filter smoothing
│   │   ├── risk_agent.py        # [BUILT] approves/rejects trades via copula risk check
│   │   ├── executor_agent.py    # [BUILT] places orders through broker layer
│   │   └── tools.py             # [BUILT] shared async DB helpers for all agents
│   ├── evolution/               # autoresearch loop
│   │   ├── evolution_engine.py  # [BUILT] karpathy loop — read/rank/kill/mutate/backtest/score/promote
│   │   ├── strategy_mutator.py  # [BUILT] claude haiku mutation + random tweak fallback
│   │   ├── report_generator.py  # [BUILT] markdown evolution reports per generation
│   │   └── documentation.py     # [BUILT] auto-updates changelog after evolution cycle
│   └── dashboard/               # [Phase 9] web dashboard
│       ├── app.py               # FastAPI backend serving analysis + trading data
│       └── templates/           # HTML templates or React frontend
├── tests/                       # test suite — 1273 tests passing
│   ├── __init__.py
│   ├── test_db.py               # [BUILT] 19 tests — pool init, migrations, masking, race conditions, env vars
│   ├── test_symbols.py          # [BUILT] 42 tests — symbol conversion, universes, resolve_universe, roundtrip
│   ├── test_resilience.py       # [BUILT] 17 tests — retry, circuit breaker, half-open, backoff, exception types
│   ├── test_validators.py       # [BUILT] 51 tests — bounds, ohlcv, fundamentals, sentiment, boundary precision, decimals
│   ├── test_market_data.py      # [BUILT] 29 tests — alpaca, yfinance, backfill, store, upsert, incremental
│   ├── test_fundamentals.py     # [BUILT] 33 tests — fetch, store, sector averages, safe_decimal, fcf_margin, upsert
│   ├── test_sec_filings.py      # [BUILT] 28 tests — insider trades, data quality, multi-type forms, fallback attrs
│   ├── test_regime_detector.py  # [BUILT] 35 tests — classify, indicators, backfill, transitions, exact calcs
│   ├── test_ratio_analysis.py   # [BUILT] 48 tests — pe/ps/peg/fcf/de exact formulas, composite weights, db
│   ├── test_dcf_model.py        # [BUILT] 23 tests — simulation, assumptions, storage, clamping, proportionality
│   ├── test_sensitivity.py      # [BUILT] 14 tests — matrix shape, monotonicity, driver detection, single cell, ties
│   ├── test_earnings_signals.py # [BUILT] 18 tests — surprise, streaks, fetch, pipeline, threshold boundary, clamping
│   ├── test_insider_activity.py # [BUILT] 29 tests — weighting, clusters, signals, db, net_buy_ratio formula, boundaries
│   ├── test_ai_summary.py       # [BUILT] 20 tests — prompt, fallback, groq mock, errors, confidence calc, cluster line
│   ├── test_trend.py            # [BUILT] 29 tests — sma/ema exact values, macd algebraic, adx ranges, supertrend flips
│   ├── test_momentum.py         # [BUILT] 31 tests — rsi bounds, stochastic at extremes, cci known, williams_r bounds, roc exact
│   ├── test_volatility.py       # [BUILT] 21 tests — bollinger exact formula/pct_b, atr convergence, keltner formula
│   ├── test_volume.py           # [BUILT] 24 tests — obv exact cumsum, vwap formula, profile poc/value_area, mfi bounds
│   ├── test_paper_broker.py     # [BUILT] 33 tests — orders, positions, balance, avg price calc, close position, limit fills
│   ├── test_alpaca_broker.py    # [BUILT] 14 tests — mocked rest api, parse_order, cancelled/partial, price validation
│   ├── test_broker_factory.py   # [BUILT] 12 tests — paper/live mode, routing, invalid mode, case sensitivity
│   ├── test_base_strategy.py    # [BUILT] 101 tests — entry/exit signals, fundamentals, position sizing, kelly exact, boundaries
│   ├── test_strategy_loader.py  # [BUILT] 97 tests — pydantic validation, config loading, track constraints, path safety, boundaries
│   ├── test_strategy_pool.py    # [BUILT] 78 tests — ranking, quartiles, lifecycle, composite exact formula, weights
│   ├── test_backtest.py         # [BUILT] 53 tests — anti-lookahead, sharpe/sortino exact, drawdown exact, edge cases
│   ├── test_monte_carlo.py      # [BUILT] 26 tests — antithetic exact negation, stratified strata coverage, control variate known answer
│   ├── test_importance_sampling.py # [BUILT] 23 tests — exponential tilt, tail P(Z>2)≈0.0228, ESS identity, optimal gamma formula
│   ├── test_risk_metrics.py     # [BUILT] 32 tests — VaR exact normal, CVaR≥VaR invariant, drawdown exact, EVT shape, annualization
│   ├── test_brier_score.py      # [BUILT] 20 tests — brier exact, decomposition identity, calibration bins, rolling window=1
│   ├── test_particle_filter.py  # [BUILT] 33 tests — bounded particles, weight invariants, convergence, collapse recovery
│   ├── test_copula_models.py    # [BUILT] 34 tests — gaussian λ=0, t-copula monotonicity, clayton formula, correlation recovery
│   ├── test_tools.py            # [BUILT] 49 tests — store/fetch helpers, whitelist, serialization roundtrips
│   ├── test_analyst_agent.py    # [BUILT] 20 tests — pipeline, partial failure, score computation, regime fetch
│   ├── test_strategy_agent.py   # [BUILT] 21 tests — signal generation, particle filter, exit checking, threshold
│   ├── test_risk_agent.py       # [BUILT] 22 tests — approve/reject, copula skip/trigger, portfolio risk, sizing
│   ├── test_executor_agent.py   # [BUILT] 15 tests — order execution, double-guard, status transitions
│   ├── test_orchestrator.py     # [BUILT] 24 tests — scheduling, market hours, startup/shutdown, error isolation
│   ├── test_strategy_mutator.py # [BUILT] 22 tests — haiku mutation, retry, random tweak, prompt construction
│   ├── test_evolution_engine.py # [BUILT] 15 tests — full loop, kill/mutate/backtest/score/promote
│   ├── test_report_generator.py # [BUILT] 10 tests — markdown format, file creation, empty sections
│   ├── test_documentation.py    # [BUILT] 8 tests — changelog insertion, content preservation
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
  "universe": "all_stocks",

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

1067 tests passing across 32 test files. Every module has tests.

### Testing Philosophy
Tests are written to **pinpoint failures**, not to pass. Every test answers a specific question:
- **Does the math match the formula?** Hand-computed reference values verified against source code formulas (e.g., PE score = (2.0-ratio)/1.5*100, Brier = mean((p-o)²), VaR = -(mu + z*sigma))
- **Do invariants hold?** Mathematical properties verified (CVaR >= VaR, particles bounded [0,1], weights sum to 1, correlation matrix positive definite, ADX in [0,100])
- **Do boundaries behave correctly?** Exact boundary values tested (pe_ratio_max at limit, max_position_pct at 0.10, max_holding_days exactly reached)
- **Do edge cases fail gracefully?** Division by zero (constant prices), empty data, NaN propagation, missing fields, negative inputs
- **Does the financial logic work?** Anti-lookahead verified (signal at previous close, fill at next open), stop loss exact trigger price, fill at better of market/limit, position weighted average cost

### Testing Patterns
- **asyncpg pool mocking**: use `_mock_pool()` helper — `MagicMock` for pool, `AsyncMock` for context manager and connection
- **external API mocking**: patch httpx.AsyncClient for alpaca, patch yfinance.Ticker for yfinance, patch edgartools for SEC
- **data validation**: test both valid and invalid inputs, verify graceful handling of edge cases
- **numpy seeded RNG**: use `np.random.default_rng(42)` for deterministic MC tests
- **hand-computed reference values**: every scoring formula, financial metric, and mathematical formula has at least one test with a hand-computed expected value
- **tight tolerances**: pytest.approx with rel=0.01 or abs=0.001 where precision matters, loose only when stochastic (MC convergence)
- **algebraic verification**: indicator outputs verified against their defining formulas (MACD = fast_ema - slow_ema, histogram = macd - signal, upper_band = middle + std_dev * rolling_std)
- **formula decomposition tests**: composite scores verified by computing each component separately and checking the weighted sum

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
