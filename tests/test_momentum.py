# / tests for momentum indicators

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.momentum import (
    StochasticResult,
    cci,
    roc,
    rsi,
    stochastic,
    williams_r,
)


def _price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    trend = np.linspace(100, 130, n)
    noise = rng.normal(0, 2, n)
    return pd.Series(trend + noise)


def _ohlc(n: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    close = np.linspace(100, 130, n) + rng.normal(0, 2, n)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return pd.Series(high), pd.Series(low), pd.Series(close)


class TestRSI:
    def test_range_0_to_100(self):
        s = _price_series(100)
        result = rsi(s)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_uptrend_high_rsi(self):
        # / strong uptrend with noise, enough points for wilder smoothing
        rng = np.random.default_rng(42)
        s = pd.Series(np.linspace(100, 200, 100) + rng.normal(0, 0.5, 100))
        result = rsi(s, period=14)
        last = result.dropna().iloc[-1]
        assert last > 70

    def test_downtrend_low_rsi(self):
        # / strong downtrend
        s = pd.Series(np.linspace(200, 100, 50))
        result = rsi(s, period=14)
        last = result.dropna().iloc[-1]
        assert last < 30

    def test_nan_for_insufficient_data(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = rsi(s, period=14)
        assert all(pd.isna(result))

    def test_flat_market_mid_rsi(self):
        # / flat market rsi should be near 50
        rng = np.random.default_rng(42)
        s = pd.Series(100 + rng.normal(0, 0.5, 100))
        result = rsi(s, period=14)
        last = result.dropna().iloc[-1]
        assert 30 < last < 70


class TestStochastic:
    def test_returns_dataclass(self):
        h, l, c = _ohlc(50)
        result = stochastic(h, l, c)
        assert isinstance(result, StochasticResult)
        assert len(result.k) == 50
        assert len(result.d) == 50

    def test_range_0_to_100(self):
        h, l, c = _ohlc(100)
        result = stochastic(h, l, c)
        valid_k = result.k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_high_at_top(self):
        # / when close is at the high end, %k should be high
        n = 30
        high = pd.Series([110.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([109.0] * n)
        result = stochastic(high, low, close, k_period=14)
        last_k = result.k.dropna().iloc[-1]
        assert last_k > 80


class TestCCI:
    def test_basic(self):
        h, l, c = _ohlc(100)
        result = cci(h, l, c)
        valid = result.dropna()
        assert len(valid) > 0

    def test_length_preserved(self):
        h, l, c = _ohlc(50)
        result = cci(h, l, c, period=20)
        assert len(result) == 50


class TestWilliamsR:
    def test_range_minus100_to_0(self):
        h, l, c = _ohlc(100)
        result = williams_r(h, l, c)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_at_high_near_zero(self):
        n = 30
        high = pd.Series([110.0] * n)
        low = pd.Series([90.0] * n)
        close = pd.Series([110.0] * n)
        result = williams_r(high, low, close, period=14)
        last = result.dropna().iloc[-1]
        assert last == pytest.approx(0.0, abs=1.0)


class TestROC:
    def test_basic(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = roc(s, period=1)
        # / 110/100 - 1 = 10%
        assert result.iloc[1] == pytest.approx(10.0)

    def test_longer_period(self):
        s = pd.Series([100.0, 105.0, 110.0, 120.0, 130.0])
        result = roc(s, period=2)
        # / 110/100 - 1 = 10%
        assert result.iloc[2] == pytest.approx(10.0)

    def test_negative_roc(self):
        s = pd.Series([100.0, 90.0])
        result = roc(s, period=1)
        assert result.iloc[1] == pytest.approx(-10.0)
