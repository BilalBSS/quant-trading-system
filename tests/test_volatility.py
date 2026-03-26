# / tests for volatility indicators

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.volatility import (
    BollingerBands,
    KeltnerChannel,
    atr,
    bollinger_bands,
    keltner_channel,
)


def _ohlc(n: int = 100, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    close = np.linspace(100, 130, n) + rng.normal(0, 2, n)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return pd.Series(high), pd.Series(low), pd.Series(close)


class TestBollingerBands:
    def test_returns_dataclass(self):
        _, _, close = _ohlc(50)
        result = bollinger_bands(close)
        assert isinstance(result, BollingerBands)

    def test_upper_above_middle_above_lower(self):
        _, _, close = _ohlc(100)
        bb = bollinger_bands(close, period=20)
        valid_idx = bb.upper.dropna().index
        assert (bb.upper[valid_idx] >= bb.middle[valid_idx]).all()
        assert (bb.middle[valid_idx] >= bb.lower[valid_idx]).all()

    def test_middle_is_sma(self):
        _, _, close = _ohlc(100)
        bb = bollinger_bands(close, period=20)
        expected_sma = close.rolling(20).mean()
        valid = bb.middle.dropna()
        np.testing.assert_allclose(valid.values, expected_sma.dropna().values, rtol=1e-10)

    def test_bandwidth_positive(self):
        _, _, close = _ohlc(100)
        bb = bollinger_bands(close)
        valid = bb.bandwidth.dropna()
        assert (valid > 0).all()

    def test_pct_b_range(self):
        _, _, close = _ohlc(100)
        bb = bollinger_bands(close)
        valid = bb.pct_b.dropna()
        # / pct_b can go outside 0-1 but most should be within
        assert valid.median() > 0
        assert valid.median() < 1

    def test_wider_std_wider_bands(self):
        _, _, close = _ohlc(100)
        narrow = bollinger_bands(close, std_dev=1.0)
        wide = bollinger_bands(close, std_dev=3.0)
        valid_idx = narrow.bandwidth.dropna().index
        assert (wide.bandwidth[valid_idx] > narrow.bandwidth[valid_idx]).all()


class TestATR:
    def test_positive(self):
        h, l, c = _ohlc(100)
        result = atr(h, l, c)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_higher_volatility_higher_atr(self):
        rng = np.random.default_rng(42)
        n = 100
        # / low volatility
        close_low = pd.Series(np.linspace(100, 110, n) + rng.normal(0, 0.5, n))
        high_low = close_low + 0.5
        low_low = close_low - 0.5
        # / high volatility
        close_high = pd.Series(np.linspace(100, 110, n) + rng.normal(0, 5, n))
        high_high = close_high + 5
        low_high = close_high - 5

        atr_low = atr(high_low, low_low, close_low).dropna().iloc[-1]
        atr_high = atr(high_high, low_high, close_high).dropna().iloc[-1]
        assert atr_high > atr_low

    def test_length_preserved(self):
        h, l, c = _ohlc(50)
        result = atr(h, l, c)
        assert len(result) == 50


class TestKeltnerChannel:
    def test_returns_dataclass(self):
        h, l, c = _ohlc(50)
        result = keltner_channel(h, l, c)
        assert isinstance(result, KeltnerChannel)

    def test_upper_above_middle_above_lower(self):
        h, l, c = _ohlc(100)
        kc = keltner_channel(h, l, c)
        valid_idx = kc.upper.dropna().index
        assert (kc.upper[valid_idx] >= kc.middle[valid_idx]).all()
        assert (kc.middle[valid_idx] >= kc.lower[valid_idx]).all()

    def test_wider_multiplier_wider_channel(self):
        h, l, c = _ohlc(100)
        narrow = keltner_channel(h, l, c, multiplier=1.0)
        wide = keltner_channel(h, l, c, multiplier=3.0)
        valid_idx = narrow.upper.dropna().index & wide.upper.dropna().index
        assert (wide.upper[valid_idx] > narrow.upper[valid_idx]).all()
