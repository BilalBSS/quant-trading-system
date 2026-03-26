# / tests for volume indicators

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.volume import (
    VolumeProfile,
    mfi,
    obv,
    volume_profile,
    vwap,
)


def _ohlcv(n: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    close = np.linspace(100, 130, n) + rng.normal(0, 2, n)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(volume)


class TestOBV:
    def test_basic_up(self):
        close = pd.Series([10.0, 11.0, 12.0])
        volume = pd.Series([100.0, 200.0, 300.0])
        result = obv(close, volume)
        # / all up days: cumulative 0 + 200 + 300 = 500
        assert result.iloc[2] == 500.0

    def test_basic_down(self):
        close = pd.Series([12.0, 11.0, 10.0])
        volume = pd.Series([100.0, 200.0, 300.0])
        result = obv(close, volume)
        # / all down: 0 - 200 - 300 = -500
        assert result.iloc[2] == -500.0

    def test_flat_no_change(self):
        close = pd.Series([10.0, 10.0, 10.0])
        volume = pd.Series([100.0, 200.0, 300.0])
        result = obv(close, volume)
        assert result.iloc[2] == 0.0

    def test_length_preserved(self):
        _, _, close, volume = _ohlcv(50)
        result = obv(close, volume)
        assert len(result) == 50


class TestVWAP:
    def test_basic(self):
        high = pd.Series([11.0, 12.0])
        low = pd.Series([9.0, 10.0])
        close = pd.Series([10.0, 11.0])
        volume = pd.Series([100.0, 100.0])
        result = vwap(high, low, close, volume)
        # / tp = (11+9+10)/3=10, (12+10+11)/3=11
        # / vwap = (10*100 + 11*100) / 200 = 10.5
        assert result.iloc[1] == pytest.approx(10.5)

    def test_higher_volume_pulls_vwap(self):
        high = pd.Series([11.0, 12.0])
        low = pd.Series([9.0, 10.0])
        close = pd.Series([10.0, 11.0])
        # / second bar has 10x volume
        volume = pd.Series([100.0, 1000.0])
        result = vwap(high, low, close, volume)
        # / vwap should be closer to bar 2's tp (11) than bar 1's tp (10)
        assert result.iloc[1] > 10.5


class TestVolumeProfile:
    def test_returns_dataclass(self):
        _, _, close, volume = _ohlcv(100)
        result = volume_profile(close, volume)
        assert isinstance(result, VolumeProfile)

    def test_poc_in_range(self):
        _, _, close, volume = _ohlcv(100)
        result = volume_profile(close, volume)
        assert result.poc >= close.min()
        assert result.poc <= close.max()

    def test_value_area(self):
        _, _, close, volume = _ohlcv(100)
        result = volume_profile(close, volume)
        assert result.value_area_low <= result.poc
        assert result.value_area_high >= result.poc

    def test_empty_series(self):
        close = pd.Series(dtype=float)
        volume = pd.Series(dtype=float)
        result = volume_profile(close, volume)
        assert result.poc == 0.0

    def test_single_price(self):
        close = pd.Series([100.0] * 10)
        volume = pd.Series([1000.0] * 10)
        result = volume_profile(close, volume)
        assert result.poc == pytest.approx(100.0)


class TestMFI:
    def test_range_0_to_100(self):
        h, l, c, v = _ohlcv(100)
        result = mfi(h, l, c, v)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_uptrend_high_mfi(self):
        # / strong uptrend with noise, enough points for wilder smoothing
        rng = np.random.default_rng(42)
        n = 100
        close = pd.Series(np.linspace(100, 200, n) + rng.normal(0, 0.5, n))
        high = close + 1
        low = close - 1
        volume = pd.Series([1_000_000.0] * n)
        result = mfi(high, low, close, volume, period=14)
        last = result.dropna().iloc[-1]
        assert last > 50

    def test_length_preserved(self):
        h, l, c, v = _ohlcv(50)
        result = mfi(h, l, c, v)
        assert len(result) == 50
