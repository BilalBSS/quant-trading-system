# / tests for trend indicators

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.trend import (
    MACDResult,
    SupertrendResult,
    adx,
    ema,
    macd,
    sma,
    supertrend,
    true_range,
)


def _price_series(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    # / trending upward with noise
    trend = np.linspace(100, 130, n)
    noise = rng.normal(0, 2, n)
    return pd.Series(trend + noise)


def _ohlc(n: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    close = np.linspace(100, 130, n) + rng.normal(0, 2, n)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return pd.Series(high), pd.Series(low), pd.Series(close)


class TestSMA:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, period=3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_nan_for_insufficient_data(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = sma(s, period=5)
        assert all(pd.isna(result))

    def test_length_preserved(self):
        s = _price_series(50)
        result = sma(s, 20)
        assert len(result) == 50


class TestEMA:
    def test_basic(self):
        s = pd.Series([10.0] * 20 + [20.0] * 20)
        result = ema(s, period=10)
        # / after 20 bars of 10, ema should be ~10
        assert result.iloc[19] == pytest.approx(10.0, abs=0.1)
        # / after 20 more bars of 20, ema should approach 20
        assert result.iloc[39] > 18.0

    def test_nan_for_insufficient_data(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = ema(s, period=5)
        assert pd.isna(result.iloc[0])

    def test_tracks_price(self):
        s = _price_series(100)
        result = ema(s, 10)
        # / ema should be close to price
        valid = result.dropna()
        assert len(valid) > 50


class TestMACD:
    def test_returns_dataclass(self):
        s = _price_series(100)
        result = macd(s)
        assert isinstance(result, MACDResult)
        assert len(result.macd_line) == 100
        assert len(result.signal_line) == 100
        assert len(result.histogram) == 100

    def test_histogram_is_diff(self):
        s = _price_series(100)
        result = macd(s)
        valid = result.histogram.dropna()
        expected = (result.macd_line - result.signal_line).dropna()
        # / last values should match
        assert valid.iloc[-1] == pytest.approx(expected.iloc[-1], abs=0.01)

    def test_uptrend_macd_positive(self):
        # / strong uptrend should give positive macd
        s = pd.Series(np.linspace(100, 200, 100))
        result = macd(s)
        # / last macd should be positive
        last_valid = result.macd_line.dropna().iloc[-1]
        assert last_valid > 0


class TestADX:
    def test_trending_market(self):
        high, low, close = _ohlc(100)
        result = adx(high, low, close, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        # / values should be positive
        assert (valid >= 0).all()

    def test_length_preserved(self):
        high, low, close = _ohlc(50)
        result = adx(high, low, close)
        assert len(result) == 50


class TestTrueRange:
    def test_basic(self):
        high = pd.Series([12.0, 11.0])
        low = pd.Series([10.0, 9.0])
        close = pd.Series([11.0, 10.0])
        result = true_range(high, low, close)
        # / first bar: high - low = 2
        assert result.iloc[0] == pytest.approx(2.0)
        # / second bar: max(11-9, |11-11|, |9-11|) = max(2, 0, 2) = 2
        assert result.iloc[1] == pytest.approx(2.0)


class TestSupertrend:
    def test_returns_dataclass(self):
        high, low, close = _ohlc(100)
        result = supertrend(high, low, close)
        assert isinstance(result, SupertrendResult)
        assert len(result.supertrend) == 100
        assert len(result.direction) == 100

    def test_direction_values(self):
        high, low, close = _ohlc(100)
        result = supertrend(high, low, close)
        valid_dirs = result.direction.dropna()
        # / direction should be 1 or -1
        assert set(valid_dirs.unique()).issubset({1, -1})

    def test_uptrend_mostly_positive(self):
        # / strong uptrend
        n = 100
        close = pd.Series(np.linspace(100, 200, n))
        high = close + 1
        low = close - 1
        result = supertrend(high, low, close)
        # / should have mostly uptrend direction toward the end
        last_20 = result.direction.iloc[-20:]
        assert (last_20 == 1).sum() > 10
