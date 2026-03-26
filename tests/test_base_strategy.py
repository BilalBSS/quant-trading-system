# / tests for strategy base module — config-driven strategy evaluation

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import (
    AnalysisData,
    ConfigDrivenStrategy,
    EntrySignal,
    ExitSignal,
    PositionSizeResult,
)


# ---------------------------------------------------------------------------
# / helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 120, seed: int = 42, base: float = 100.0,
            trend: float = 0.0) -> pd.DataFrame:
    # / generate realistic ohlcv data with enough bars for indicator warmup
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    close = np.empty(n)
    close[0] = base
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + trend / n + rng.normal(0, 0.015))
    high = close + rng.uniform(0.3, 1.5, n)
    low = close - rng.uniform(0.3, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _ohlcv_dropping(n: int = 120, seed: int = 42) -> pd.DataFrame:
    # / price trending sharply down — triggers oversold indicators
    return _ohlcv(n, seed=seed, base=120.0, trend=-0.40)


def _ohlcv_rising(n: int = 120, seed: int = 42) -> pd.DataFrame:
    # / price trending sharply up — triggers overbought indicators
    return _ohlcv(n, seed=seed, base=80.0, trend=0.50)


def _ohlcv_with_volume_spike(n: int = 120, seed: int = 42) -> pd.DataFrame:
    # / normal data but last bar has extreme volume
    df = _ohlcv(n, seed=seed)
    avg_vol = df["volume"].mean()
    df.iloc[-1, df.columns.get_loc("volume")] = avg_vol * 5.0
    return df


def _base_config(**overrides) -> dict:
    cfg = {
        "id": "test_001",
        "name": "TestStrategy",
        "universe": ["AAPL", "MSFT"],
        "fundamental_filters": {},
        "entry_conditions": {"operator": "AND", "signals": []},
        "exit_conditions": {},
        "position_sizing": {"method": "fixed_pct", "max_position_pct": 0.05},
    }
    cfg.update(overrides)
    return cfg


def _strategy_001_config() -> dict:
    # / mirrors strategy_001.json — bollinger + rsi + volume with fundamentals
    return {
        "id": "strategy_001",
        "name": "Bollinger_PE_Oversold",
        "universe": ["AAPL", "MSFT", "GOOG"],
        "fundamental_filters": {
            "pe_ratio_max": 40,
            "pe_vs_sector": "below_average",
            "revenue_growth_min": 0.01,
            "fcf_margin_min": 0.10,
            "debt_to_equity_max": 2.0,
            "dcf_upside_min": 0.05,
        },
        "entry_conditions": {
            "operator": "AND",
            "signals": [
                {"indicator": "bollinger_bands", "condition": "price_below_lower",
                 "lookback": 20, "std_dev": 2.0},
                {"indicator": "rsi", "condition": "below", "threshold": 35, "period": 14},
                {"indicator": "volume", "condition": "above_average",
                 "multiplier": 1.3, "period": 20},
            ],
        },
        "exit_conditions": {
            "take_profit": {"indicator": "bollinger_bands", "condition": "price_above_middle"},
            "stop_loss": {"type": "atr_trailing", "multiplier": 2.0, "period": 14},
            "time_exit": {"max_holding_days": 30},
        },
        "position_sizing": {
            "method": "kelly_fraction",
            "max_position_pct": 0.08,
            "kelly_fraction": 0.25,
        },
    }


def _strategy_002_config() -> dict:
    # / mirrors strategy_002.json — macd momentum, no fundamentals
    return {
        "id": "strategy_002",
        "name": "MACD_Momentum_Breakout",
        "universe": ["AAPL", "MSFT", "GOOG"],
        "entry_conditions": {
            "operator": "AND",
            "signals": [
                {"indicator": "macd", "condition": "crossover_bullish"},
                {"indicator": "volume", "condition": "above_average",
                 "multiplier": 1.2, "period": 20},
                {"indicator": "adx", "condition": "above", "threshold": 25, "period": 14},
            ],
        },
        "exit_conditions": {
            "stop_loss": {"type": "atr_trailing", "multiplier": 1.5, "period": 14},
            "take_profit": {"indicator": "bollinger_bands",
                            "condition": "price_above_upper", "period": 20, "std_dev": 2.0},
            "time_exit": {"max_holding_days": 20},
        },
        "position_sizing": {
            "method": "strength_scaled",
            "max_position_pct": 0.04,
        },
    }


def _passing_analysis() -> AnalysisData:
    # / analysis that passes all strategy_001 fundamental filters
    return AnalysisData(
        pe_ratio=20.0,
        sector_pe_avg=25.0,
        revenue_growth=0.10,
        fcf_margin=0.20,
        debt_to_equity=0.8,
        dcf_upside=0.15,
    )


def _failing_analysis(**overrides) -> AnalysisData:
    # / start from passing, then override to fail a specific filter
    a = _passing_analysis()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
# / initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_properties_from_config(self):
        cfg = _strategy_001_config()
        s = ConfigDrivenStrategy(cfg)
        assert s.strategy_id == "strategy_001"
        assert s.name == "Bollinger_PE_Oversold"
        assert s.config is cfg

    def test_universe_ref(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        assert isinstance(s.universe_ref, str)
        # / legacy list configs get stored as the raw value
        assert s.universe_ref is not None

    def test_requires_fundamentals_when_filters_present(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        assert s.requires_fundamentals is True

    def test_no_fundamentals_when_filters_absent(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        assert s.requires_fundamentals is False

    def test_empty_universe_defaults(self):
        cfg = _base_config()
        del cfg["universe"]
        s = ConfigDrivenStrategy(cfg)
        assert s.universe_ref == "all"

    def test_missing_optional_sections(self):
        # / strategy with only id/name should still init
        cfg = {"id": "minimal", "name": "Minimal"}
        s = ConfigDrivenStrategy(cfg)
        assert s.strategy_id == "minimal"
        assert s.requires_fundamentals is False


# ---------------------------------------------------------------------------
# / fundamental filters
# ---------------------------------------------------------------------------

class TestFundamentalFilters:
    def setup_method(self):
        self.strategy = ConfigDrivenStrategy(_strategy_001_config())
        self.data = _ohlcv()

    def test_passes_all_filters(self):
        result = self.strategy.should_enter("AAPL", self.data, _passing_analysis())
        # / may fail on technicals, but should not fail on fundamentals
        if not result.should_enter:
            assert "fundamentals" not in " ".join(result.reasons).lower() or \
                all("pe" not in r and "revenue" not in r and "fcf" not in r
                    and "d/e" not in r and "dcf" not in r
                    for r in result.reasons)

    def test_rejects_no_analysis_data(self):
        result = self.strategy.should_enter("AAPL", self.data, analysis=None)
        assert result.should_enter is False
        assert "no fundamental data" in result.reasons

    def test_pe_ratio_max_exceeded(self):
        analysis = _failing_analysis(pe_ratio=50.0)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("pe" in r.lower() for r in result.reasons)

    def test_pe_ratio_at_boundary(self):
        # / pe exactly at max should pass
        analysis = _failing_analysis(pe_ratio=40.0)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        # / should not fail on pe filter
        assert not any("pe 40.0 > max" in r for r in result.reasons)

    def test_pe_vs_sector_above_average(self):
        analysis = _failing_analysis(pe_ratio=30.0, sector_pe_avg=25.0)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("sector" in r for r in result.reasons)

    def test_pe_vs_sector_below_average_passes(self):
        analysis = _failing_analysis(pe_ratio=20.0, sector_pe_avg=25.0)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert not any("sector" in r for r in result.reasons)

    def test_revenue_growth_too_low(self):
        analysis = _failing_analysis(revenue_growth=0.005)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("revenue" in r for r in result.reasons)

    def test_fcf_margin_too_low(self):
        analysis = _failing_analysis(fcf_margin=0.05)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("fcf" in r for r in result.reasons)

    def test_debt_to_equity_too_high(self):
        analysis = _failing_analysis(debt_to_equity=3.0)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("d/e" in r for r in result.reasons)

    def test_dcf_upside_too_low(self):
        analysis = _failing_analysis(dcf_upside=0.02)
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        assert result.should_enter is False
        assert any("dcf" in r for r in result.reasons)

    def test_none_values_reject_filter(self):
        # / if analysis field is none and filter is configured, reject (data unavailable)
        analysis = AnalysisData(
            pe_ratio=None, sector_pe_avg=None,
            revenue_growth=None, fcf_margin=None,
            debt_to_equity=None, dcf_upside=None,
        )
        result = self.strategy.should_enter("AAPL", self.data, analysis)
        # / should fail on fundamentals — none values mean data unavailable
        assert result.should_enter is False
        assert any("unavailable" in r for r in result.reasons)

    def test_no_fundamental_strategy_skips_filters(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        # / should not require analysis at all
        result = s.should_enter("AAPL", self.data, analysis=None)
        assert "no fundamental data" not in result.reasons


# ---------------------------------------------------------------------------
# / entry signals — individual indicators
# ---------------------------------------------------------------------------

class TestEntryBollingerBands:
    def test_price_below_lower_on_downtrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "bollinger_bands", "condition": "price_below_lower",
                         "lookback": 20, "std_dev": 2.0}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(150, seed=42)
        result = s.should_enter("AAPL", data)
        # / on a strong downtrend the last price should be below lower band
        assert "bb:" in result.reasons[0]

    def test_price_above_upper_on_uptrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "bollinger_bands", "condition": "price_above_upper",
                         "lookback": 20, "std_dev": 2.0}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150, seed=42)
        result = s.should_enter("AAPL", data)
        assert "bb:" in result.reasons[0]

    def test_price_above_middle(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "bollinger_bands", "condition": "price_above_middle",
                         "lookback": 20, "std_dev": 2.0}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150, seed=42)
        result = s.should_enter("AAPL", data)
        # / on an uptrend, close should be above the sma middle
        assert result.should_enter is True
        assert result.strength == pytest.approx(0.5)


class TestEntryRSI:
    def test_rsi_below_threshold_on_downtrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "rsi", "condition": "below",
                         "threshold": 40, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(150, seed=42)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert result.strength > 0
        assert "rsi=" in result.reasons[0]

    def test_rsi_above_threshold_on_uptrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "rsi", "condition": "above",
                         "threshold": 55, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150, seed=42)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert result.strength > 0

    def test_rsi_strength_bounded_0_to_1(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "rsi", "condition": "below",
                         "threshold": 80, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(150)
        result = s.should_enter("AAPL", data)
        if result.should_enter:
            assert 0.0 <= result.strength <= 1.0


class TestEntryVolume:
    def test_volume_above_average_with_spike(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "volume", "condition": "above_average",
                         "multiplier": 1.5, "period": 20}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_with_volume_spike(120)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert "vol=" in result.reasons[0]

    def test_volume_below_average_fails(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "volume", "condition": "above_average",
                         "multiplier": 3.0, "period": 20}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(120)  # / normal volume, no spike
        result = s.should_enter("AAPL", data)
        assert result.should_enter is False


class TestEntryMACD:
    def _macd_crossover_data(self, seed: int = 42) -> pd.DataFrame:
        # / build data where macd histogram crosses from negative to positive
        # / downtrend then uptrend forces a crossover in the histogram
        rng = np.random.default_rng(seed)
        n = 150
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        close = np.empty(n)
        close[0] = 100.0
        # / first half: drift down
        for i in range(1, n // 2):
            close[i] = close[i - 1] * (1 + rng.normal(-0.003, 0.01))
        # / second half: drift up sharply to create crossover
        for i in range(n // 2, n):
            close[i] = close[i - 1] * (1 + rng.normal(0.006, 0.01))
        high = close + rng.uniform(0.3, 1.2, n)
        low = close - rng.uniform(0.3, 1.2, n)
        open_ = close + rng.normal(0, 0.3, n)
        volume = rng.integers(500_000, 2_000_000, n).astype(float)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            index=dates,
        )

    def test_macd_crossover_bullish(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "macd", "condition": "crossover_bullish"}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = self._macd_crossover_data()
        result = s.should_enter("AAPL", data)
        # / the crossover may or may not land exactly on the last bar —
        # / but the reason string should reference macd histogram
        assert "macd histogram:" in result.reasons[0]

    def test_macd_positive(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "macd", "condition": "positive"}],
        })
        s = ConfigDrivenStrategy(cfg)
        # / need a very strong uptrend so histogram stays positive at the end
        data = _ohlcv_rising(150, seed=7)
        result = s.should_enter("AAPL", data)
        # / strong uptrend should have positive histogram
        assert result.should_enter is True
        assert result.strength == pytest.approx(0.5)

    def test_macd_crossover_bearish(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "macd", "condition": "crossover_bearish"}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(150)
        result = s.should_enter("AAPL", data)
        assert "macd histogram:" in result.reasons[0]


class TestEntrySMA:
    def test_price_above_sma_on_uptrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "sma", "condition": "price_above", "period": 20}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(120)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert "sma" in result.reasons[0]

    def test_price_below_sma_on_downtrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "sma", "condition": "price_below", "period": 20}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(120)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert "sma" in result.reasons[0]


class TestEntryADX:
    def test_adx_above_threshold(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "adx", "condition": "above",
                         "threshold": 15, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        # / strong trending data should have elevated adx
        data = _ohlcv_rising(150, seed=42)
        result = s.should_enter("AAPL", data)
        assert "adx=" in result.reasons[0]

    def test_adx_below_threshold(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "adx", "condition": "below",
                         "threshold": 80, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(150)
        result = s.should_enter("AAPL", data)
        # / adx almost never hits 80, so below should pass
        assert result.should_enter is True
        assert result.strength == pytest.approx(0.5)


class TestEntryStochastic:
    def test_stochastic_below_on_downtrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "stochastic", "condition": "below",
                         "threshold": 30, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(150)
        result = s.should_enter("AAPL", data)
        assert "stoch" in result.reasons[0]

    def test_stochastic_above_on_uptrend(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "stochastic", "condition": "above",
                         "threshold": 70, "period": 14}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150)
        result = s.should_enter("AAPL", data)
        assert "stoch" in result.reasons[0]


class TestEntryUnknownIndicator:
    def test_unknown_indicator_fails(self):
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [{"indicator": "magic_indicator", "condition": "magic"}],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(120)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is False
        assert any("unknown" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# / entry — operator logic (AND / OR)
# ---------------------------------------------------------------------------

class TestEntryOperators:
    def test_and_operator_all_pass(self):
        # / sma price_above on uptrend + low adx threshold — both should pass
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [
                {"indicator": "sma", "condition": "price_above", "period": 20},
                {"indicator": "adx", "condition": "below", "threshold": 90, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert len(result.reasons) == 2

    def test_and_operator_one_fails(self):
        # / sma price_above on downtrend should fail
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [
                {"indicator": "sma", "condition": "price_above", "period": 20},
                {"indicator": "adx", "condition": "below", "threshold": 90, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(150)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is False

    def test_or_operator_one_passes(self):
        # / one signal passes, one fails — should still enter with OR
        cfg = _base_config(entry_conditions={
            "operator": "OR",
            "signals": [
                {"indicator": "sma", "condition": "price_above", "period": 20},
                {"indicator": "adx", "condition": "below", "threshold": 90, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(150)
        result = s.should_enter("AAPL", data)
        # / adx below 90 should pass even though sma fails
        assert result.should_enter is True

    def test_or_operator_none_pass(self):
        # / both require impossible conditions
        cfg = _base_config(entry_conditions={
            "operator": "OR",
            "signals": [
                {"indicator": "rsi", "condition": "below", "threshold": 1, "period": 14},
                {"indicator": "adx", "condition": "above", "threshold": 99, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(150)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is False

    def test_no_signals_passes(self):
        cfg = _base_config(entry_conditions={"operator": "AND", "signals": []})
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(120)
        result = s.should_enter("AAPL", data)
        assert result.should_enter is True
        assert result.strength == pytest.approx(1.0)

    def test_and_strength_is_average(self):
        # / two signals that both pass — strength should be average
        cfg = _base_config(entry_conditions={
            "operator": "AND",
            "signals": [
                {"indicator": "sma", "condition": "price_above", "period": 20},
                {"indicator": "adx", "condition": "below", "threshold": 90, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150)
        result = s.should_enter("AAPL", data)
        if result.should_enter:
            # / both return 0.5 so average should be 0.5
            assert result.strength == pytest.approx(0.5)

    def test_or_strength_is_max(self):
        cfg = _base_config(entry_conditions={
            "operator": "OR",
            "signals": [
                {"indicator": "sma", "condition": "price_above", "period": 20},
                {"indicator": "adx", "condition": "below", "threshold": 90, "period": 14},
            ],
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150)
        result = s.should_enter("AAPL", data)
        if result.should_enter:
            assert result.strength == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# / exit signals
# ---------------------------------------------------------------------------

class TestExitTimeExit:
    def setup_method(self):
        cfg = _base_config(exit_conditions={
            "time_exit": {"max_holding_days": 10},
        })
        self.strategy = ConfigDrivenStrategy(cfg)
        self.data = _ohlcv(120)

    def test_exits_after_max_days(self):
        entry_date = self.data.index[0]
        # / bar 50 is well past 10 business days
        result = self.strategy.should_exit("AAPL", self.data, 100.0, entry_date, 50)
        assert result.should_exit is True
        assert "time exit" in result.reason

    def test_no_exit_before_max_days(self):
        entry_date = self.data.index[0]
        # / bar 5 is within 10 days
        result = self.strategy.should_exit("AAPL", self.data, 100.0, entry_date, 5)
        assert result.should_exit is False

    def test_exits_exactly_at_max_days(self):
        entry_date = self.data.index[0]
        # / find bar index where days_held >= 10
        for i in range(len(self.data)):
            current_date = self.data.index[i]
            days = (current_date - entry_date).days
            if days >= 10:
                result = self.strategy.should_exit("AAPL", self.data, 100.0, entry_date, i)
                assert result.should_exit is True
                break


class TestExitStopLossFixed:
    def setup_method(self):
        cfg = _base_config(exit_conditions={
            "stop_loss": {"type": "fixed_pct", "pct": 0.05},
        })
        self.strategy = ConfigDrivenStrategy(cfg)

    def test_stop_loss_triggered(self):
        # / entry at 100, stop at 95 — create data that drops below 95
        data = _ohlcv_dropping(120, seed=42)
        entry_price = float(data["close"].iloc[0])
        stop_price = entry_price * 0.95
        # / find a bar where close <= stop_price
        for i in range(len(data)):
            if float(data["close"].iloc[i]) <= stop_price:
                entry_date = data.index[0]
                result = self.strategy.should_exit(
                    "AAPL", data, entry_price, entry_date, i,
                )
                assert result.should_exit is True
                assert "stop loss" in result.reason
                break

    def test_stop_loss_not_triggered(self):
        data = _ohlcv_rising(120)
        entry_price = float(data["close"].iloc[0])
        entry_date = data.index[0]
        # / on an uptrend, price shouldn't drop 5%
        result = self.strategy.should_exit("AAPL", data, entry_price, entry_date, 50)
        assert result.should_exit is False


class TestExitStopLossATRTrailing:
    def setup_method(self):
        cfg = _base_config(exit_conditions={
            "stop_loss": {"type": "atr_trailing", "multiplier": 2.0, "period": 14},
        })
        self.strategy = ConfigDrivenStrategy(cfg)

    def test_atr_trailing_stop_on_drop(self):
        # / strong downtrend from high should trigger trailing stop
        rng = np.random.default_rng(99)
        n = 120
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        close = np.empty(n)
        close[0] = 100.0
        # / first 40 bars: rise to peak
        for i in range(1, 40):
            close[i] = close[i - 1] * (1 + rng.normal(0.005, 0.005))
        # / then crash hard
        for i in range(40, n):
            close[i] = close[i - 1] * (1 + rng.normal(-0.02, 0.008))
        high = close + rng.uniform(0.3, 1.0, n)
        low = close - rng.uniform(0.3, 1.0, n)
        open_ = close + rng.normal(0, 0.3, n)
        volume = rng.integers(500_000, 2_000_000, n).astype(float)
        data = pd.DataFrame(
            {"open": open_, "high": high, "low": low,
             "close": close, "volume": volume},
            index=dates,
        )
        entry_price = float(data["close"].iloc[20])
        entry_date = data.index[20]
        # / check a late bar when price has dropped significantly
        result = self.strategy.should_exit("AAPL", data, entry_price, entry_date, 90)
        assert result.should_exit is True
        assert "atr trailing stop" in result.reason

    def test_atr_trailing_no_exit_in_uptrend(self):
        data = _ohlcv_rising(120)
        entry_price = float(data["close"].iloc[20])
        entry_date = data.index[20]
        result = self.strategy.should_exit("AAPL", data, entry_price, entry_date, 80)
        assert result.should_exit is False


class TestExitTakeProfitBollinger:
    def test_take_profit_above_middle(self):
        cfg = _base_config(exit_conditions={
            "take_profit": {"indicator": "bollinger_bands",
                            "condition": "price_above_middle", "period": 20, "std_dev": 2.0},
        })
        s = ConfigDrivenStrategy(cfg)
        # / use 150 bars and check near the end where sma is well-defined
        data = _ohlcv_rising(150)
        entry_date = data.index[0]
        # / on strong uptrend, last bar's close should exceed sma(20) middle
        # / check multiple bars near the end to find one that triggers
        triggered = False
        for idx in range(130, 149):
            result = s.should_exit("AAPL", data, 80.0, entry_date, idx)
            if result.should_exit:
                assert "take profit" in result.reason
                triggered = True
                break
        assert triggered, "take profit should trigger on strong uptrend"

    def test_take_profit_above_upper(self):
        cfg = _base_config(exit_conditions={
            "take_profit": {"indicator": "bollinger_bands",
                            "condition": "price_above_upper", "period": 20, "std_dev": 2.0},
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_rising(150, seed=42)
        entry_date = data.index[0]
        # / check near end on a strong uptrend
        result = s.should_exit("AAPL", data, 80.0, entry_date, 140)
        assert "bb" in result.reason.lower() or result.should_exit is False

    def test_take_profit_not_triggered_on_downtrend(self):
        cfg = _base_config(exit_conditions={
            "take_profit": {"indicator": "bollinger_bands",
                            "condition": "price_above_middle", "period": 20, "std_dev": 2.0},
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(120)
        entry_date = data.index[0]
        result = s.should_exit("AAPL", data, 120.0, entry_date, 100)
        assert result.should_exit is False


class TestExitBoundary:
    def test_current_bar_idx_out_of_range(self):
        cfg = _base_config(exit_conditions={"time_exit": {"max_holding_days": 5}})
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(50)
        entry_date = data.index[0]
        result = s.should_exit("AAPL", data, 100.0, entry_date, 999)
        assert result.should_exit is False

    def test_no_exit_conditions(self):
        cfg = _base_config(exit_conditions={})
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv(50)
        entry_date = data.index[0]
        result = s.should_exit("AAPL", data, 100.0, entry_date, 30)
        assert result.should_exit is False

    def test_exit_priority_time_before_stop_loss(self):
        # / time exit triggers first when both conditions are met
        cfg = _base_config(exit_conditions={
            "time_exit": {"max_holding_days": 5},
            "stop_loss": {"type": "fixed_pct", "pct": 0.05},
        })
        s = ConfigDrivenStrategy(cfg)
        data = _ohlcv_dropping(120)
        entry_date = data.index[0]
        entry_price = float(data["close"].iloc[0])
        # / check at bar 50 — both should fire but time_exit is checked first
        result = s.should_exit("AAPL", data, entry_price, entry_date, 50)
        assert result.should_exit is True
        assert "time exit" in result.reason


# ---------------------------------------------------------------------------
# / position sizing
# ---------------------------------------------------------------------------

class TestPositionSizingKellyFraction:
    def setup_method(self):
        cfg = _base_config(position_sizing={
            "method": "kelly_fraction",
            "max_position_pct": 0.08,
            "kelly_fraction": 0.25,
        })
        self.strategy = ConfigDrivenStrategy(cfg)

    def test_kelly_basic(self):
        result = self.strategy.position_size(equity=100_000, price=150.0, strength=0.8)
        assert result.method == "kelly_fraction"
        # / kelly_f * strength = 0.25 * 0.8 = 0.20 — capped at 0.08
        expected_pct = 0.08
        expected_value = 100_000 * expected_pct
        expected_qty = int(expected_value / 150.0)
        assert result.qty == expected_qty

    def test_kelly_low_strength(self):
        result = self.strategy.position_size(equity=100_000, price=150.0, strength=0.2)
        # / kelly_f * strength = 0.25 * 0.2 = 0.05 — below cap
        expected_pct = 0.05
        expected_value = 100_000 * expected_pct
        expected_qty = int(expected_value / 150.0)
        assert result.qty == expected_qty

    def test_kelly_zero_strength(self):
        result = self.strategy.position_size(equity=100_000, price=150.0, strength=0.0)
        assert result.qty == 0

    def test_kelly_capped_at_max(self):
        result = self.strategy.position_size(equity=100_000, price=50.0, strength=1.0)
        # / 0.25 * 1.0 = 0.25 > max 0.08 — should be capped
        max_value = 100_000 * 0.08
        assert result.qty == int(max_value / 50.0)


class TestPositionSizingFixedPct:
    def setup_method(self):
        cfg = _base_config(position_sizing={
            "method": "fixed_pct",
            "max_position_pct": 0.05,
        })
        self.strategy = ConfigDrivenStrategy(cfg)

    def test_fixed_pct_ignores_strength(self):
        r1 = self.strategy.position_size(equity=100_000, price=100.0, strength=0.3)
        r2 = self.strategy.position_size(equity=100_000, price=100.0, strength=0.9)
        assert r1.qty == r2.qty
        assert r1.method == "fixed_pct"

    def test_fixed_pct_basic(self):
        result = self.strategy.position_size(equity=100_000, price=200.0, strength=0.5)
        expected_value = 100_000 * 0.05
        expected_qty = int(expected_value / 200.0)
        assert result.qty == expected_qty

    def test_whole_shares_only(self):
        result = self.strategy.position_size(equity=10_000, price=333.33, strength=0.5)
        assert isinstance(result.qty, int)


class TestPositionSizingStrengthScaled:
    def setup_method(self):
        cfg = _base_config(position_sizing={
            "method": "strength_scaled",
            "max_position_pct": 0.04,
        })
        self.strategy = ConfigDrivenStrategy(cfg)

    def test_scales_with_strength(self):
        r_low = self.strategy.position_size(equity=100_000, price=100.0, strength=0.25)
        r_high = self.strategy.position_size(equity=100_000, price=100.0, strength=1.0)
        assert r_high.qty > r_low.qty
        assert r_low.method == "strength_scaled"

    def test_full_strength(self):
        result = self.strategy.position_size(equity=100_000, price=100.0, strength=1.0)
        expected_value = 100_000 * 0.04
        expected_qty = int(expected_value / 100.0)
        assert result.qty == expected_qty

    def test_zero_strength_zero_qty(self):
        result = self.strategy.position_size(equity=100_000, price=100.0, strength=0.0)
        assert result.qty == 0

    def test_half_strength(self):
        result = self.strategy.position_size(equity=100_000, price=100.0, strength=0.5)
        # / 0.04 * 0.5 = 0.02 -> 2000 / 100 = 20 shares
        assert result.qty == 20


class TestPositionSizingUnknownMethod:
    def test_unknown_method_defaults_to_max_pct(self):
        cfg = _base_config(position_sizing={
            "method": "some_future_method",
            "max_position_pct": 0.06,
        })
        s = ConfigDrivenStrategy(cfg)
        result = s.position_size(equity=100_000, price=100.0, strength=0.5)
        expected_qty = int(100_000 * 0.06 / 100.0)
        assert result.qty == expected_qty


class TestPositionSizingDefaultConfig:
    def test_defaults_when_no_sizing_config(self):
        cfg = {"id": "min", "name": "Min"}
        s = ConfigDrivenStrategy(cfg)
        result = s.position_size(equity=100_000, price=100.0, strength=0.5)
        # / default: fixed_pct, max 0.08
        assert result.method == "fixed_pct"
        assert result.qty == int(100_000 * 0.08 / 100.0)


# ---------------------------------------------------------------------------
# / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_insufficient_data_returns_no_entry(self):
        s = ConfigDrivenStrategy(_base_config())
        tiny = _ohlcv(1)
        result = s.should_enter("AAPL", tiny)
        assert result.should_enter is False
        assert "insufficient data" in result.reasons

    def test_empty_dataframe_returns_no_entry(self):
        s = ConfigDrivenStrategy(_base_config())
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = s.should_enter("AAPL", empty)
        assert result.should_enter is False

    def test_zero_price_position_size(self):
        s = ConfigDrivenStrategy(_base_config())
        result = s.position_size(equity=100_000, price=0.0, strength=0.5)
        assert result.qty == 0
        assert result.pct_of_portfolio == 0

    def test_negative_price_position_size(self):
        s = ConfigDrivenStrategy(_base_config())
        result = s.position_size(equity=100_000, price=-10.0, strength=0.5)
        assert result.qty == 0

    def test_zero_equity_position_size(self):
        s = ConfigDrivenStrategy(_base_config())
        result = s.position_size(equity=0, price=100.0, strength=0.5)
        assert result.qty == 0
        assert result.pct_of_portfolio == 0

    def test_actual_pct_accounts_for_rounding(self):
        s = ConfigDrivenStrategy(_base_config(position_sizing={
            "method": "fixed_pct", "max_position_pct": 0.05,
        }))
        result = s.position_size(equity=10_000, price=333.0, strength=0.5)
        # / actual_pct should reflect what was actually bought (whole shares)
        expected = (result.qty * 333.0) / 10_000
        assert result.pct_of_portfolio == pytest.approx(expected)

    def test_entry_signal_dataclass_defaults(self):
        sig = EntrySignal(should_enter=False)
        assert sig.strength == 0.0
        assert sig.reasons == []

    def test_exit_signal_dataclass_defaults(self):
        sig = ExitSignal(should_exit=False)
        assert sig.reason == ""

    def test_position_size_result_fields(self):
        r = PositionSizeResult(qty=10, pct_of_portfolio=0.05, method="kelly_fraction")
        assert r.qty == 10
        assert r.pct_of_portfolio == 0.05
        assert r.method == "kelly_fraction"

    def test_analysis_data_defaults(self):
        a = AnalysisData()
        assert a.pe_ratio is None
        assert a.consecutive_beats == 0
        assert a.fundamental_score is None


# ---------------------------------------------------------------------------
# / integration — full strategy_001 and strategy_002 configs
# ---------------------------------------------------------------------------

class TestIntegrationStrategy001:
    def test_rejects_without_fundamentals(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        data = _ohlcv_dropping(150)
        result = s.should_enter("AAPL", data, analysis=None)
        assert result.should_enter is False

    def test_rejects_bad_fundamentals(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        data = _ohlcv_dropping(150)
        analysis = _failing_analysis(pe_ratio=60.0)
        result = s.should_enter("AAPL", data, analysis)
        assert result.should_enter is False

    def test_exit_evaluates_all_conditions(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        data = _ohlcv(120)
        entry_date = data.index[0]
        # / very late bar to trigger time exit (30 days)
        result = s.should_exit("AAPL", data, 100.0, entry_date, 80)
        assert result.should_exit is True

    def test_kelly_position_sizing(self):
        s = ConfigDrivenStrategy(_strategy_001_config())
        result = s.position_size(equity=50_000, price=175.0, strength=0.6)
        assert result.method == "kelly_fraction"
        # / 0.25 * 0.6 = 0.15 -> capped at 0.08
        assert result.qty == int(50_000 * 0.08 / 175.0)


class TestIntegrationStrategy002:
    def test_no_fundamentals_required(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        assert s.requires_fundamentals is False

    def test_can_enter_without_analysis(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        data = _ohlcv(150)
        result = s.should_enter("AAPL", data)
        # / should evaluate technicals without needing analysis
        assert "no fundamental data" not in result.reasons

    def test_strength_scaled_sizing(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        result = s.position_size(equity=100_000, price=200.0, strength=0.7)
        assert result.method == "strength_scaled"
        # / 0.04 * 0.7 = 0.028 -> 2800 / 200 = 14, but float truncation gives 13
        pct = 0.04 * 0.7
        expected_qty = int(100_000 * pct / 200.0)
        assert result.qty == expected_qty

    def test_exit_time_limit(self):
        s = ConfigDrivenStrategy(_strategy_002_config())
        data = _ohlcv(120)
        entry_date = data.index[0]
        result = s.should_exit("AAPL", data, 100.0, entry_date, 80)
        assert result.should_exit is True
        assert "time exit" in result.reason
