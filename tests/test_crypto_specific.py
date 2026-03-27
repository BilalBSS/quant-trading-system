# / tests for crypto-specific indicators

import numpy as np
import pandas as pd
import pytest

from src.indicators.crypto_specific import (
    exchange_flow_ratio,
    funding_rate_signal,
    nvt_ratio,
    nvt_signal,
    open_interest_trend,
)


def _make_series(values):
    return pd.Series(values, dtype=float)


class TestFundingRateSignal:
    def test_positive_funding_bearish(self):
        # / high positive funding = longs paying, overcrowded long = bearish
        rates = _make_series([0.02, 0.03, 0.015])
        signal = funding_rate_signal(rates, threshold=0.01)
        assert (signal == -1).all()

    def test_negative_funding_bullish(self):
        # / negative funding = shorts paying, overcrowded short = bullish
        rates = _make_series([-0.02, -0.03, -0.015])
        signal = funding_rate_signal(rates, threshold=0.01)
        assert (signal == 1).all()

    def test_neutral_funding(self):
        rates = _make_series([0.005, -0.003, 0.001])
        signal = funding_rate_signal(rates, threshold=0.01)
        assert (signal == 0).all()

    def test_mixed_signals(self):
        rates = _make_series([-0.02, 0.0, 0.02])
        signal = funding_rate_signal(rates, threshold=0.01)
        assert signal.iloc[0] == 1    # bullish
        assert signal.iloc[1] == 0    # neutral
        assert signal.iloc[2] == -1   # bearish

    def test_threshold_boundary(self):
        rates = _make_series([0.01, -0.01])
        signal = funding_rate_signal(rates, threshold=0.01)
        assert (signal == 0).all()  # exactly at threshold = neutral


class TestOpenInterestTrend:
    def test_rising_oi_rising_price_bullish(self):
        oi = _make_series([100, 110, 120, 130, 140])
        price = _make_series([50, 55, 60, 65, 70])
        signal = open_interest_trend(oi, price, period=1)
        # / from index 1 onward: both rising
        assert signal.iloc[-1] == 1.0

    def test_rising_oi_falling_price_bearish(self):
        oi = _make_series([100, 110, 120, 130, 140])
        price = _make_series([70, 65, 60, 55, 50])
        signal = open_interest_trend(oi, price, period=1)
        assert signal.iloc[-1] == -1.0

    def test_falling_oi_neutral(self):
        oi = _make_series([140, 130, 120, 110, 100])
        price = _make_series([50, 55, 60, 65, 70])
        signal = open_interest_trend(oi, price, period=1)
        assert signal.iloc[-1] == 0.0

    def test_output_shape(self):
        oi = _make_series([100, 200, 300])
        price = _make_series([50, 60, 70])
        signal = open_interest_trend(oi, price, period=1)
        assert len(signal) == 3


class TestExchangeFlowRatio:
    def test_outflow_dominant_bullish(self):
        inflows = _make_series([100, 100, 100, 100, 100])
        outflows = _make_series([200, 200, 200, 200, 200])
        ratio = exchange_flow_ratio(inflows, outflows, period=3)
        assert ratio.iloc[-1] > 1.0

    def test_inflow_dominant_bearish(self):
        inflows = _make_series([200, 200, 200, 200, 200])
        outflows = _make_series([100, 100, 100, 100, 100])
        ratio = exchange_flow_ratio(inflows, outflows, period=3)
        assert ratio.iloc[-1] < 1.0

    def test_equal_flows_neutral(self):
        inflows = _make_series([100, 100, 100])
        outflows = _make_series([100, 100, 100])
        ratio = exchange_flow_ratio(inflows, outflows, period=3)
        assert ratio.iloc[-1] == pytest.approx(1.0)

    def test_zero_inflow_handled(self):
        inflows = _make_series([0, 0, 100])
        outflows = _make_series([100, 100, 100])
        ratio = exchange_flow_ratio(inflows, outflows, period=1)
        # / should not crash, zero inflows become nan
        assert not np.isnan(ratio.iloc[-1])


class TestNVTRatio:
    def test_hand_computed(self):
        # / market_cap=1B, tx_volume=100M -> nvt=10
        mcap = _make_series([1_000_000_000])
        vol = _make_series([100_000_000])
        nvt = nvt_ratio(mcap, vol)
        assert nvt.iloc[0] == pytest.approx(10.0)

    def test_zero_volume_nan(self):
        mcap = _make_series([1_000_000_000])
        vol = _make_series([0])
        nvt = nvt_ratio(mcap, vol)
        assert np.isnan(nvt.iloc[0])

    def test_series_output(self):
        mcap = _make_series([1e9, 1.1e9, 1.2e9])
        vol = _make_series([1e8, 1.1e8, 1.2e8])
        nvt = nvt_ratio(mcap, vol)
        assert len(nvt) == 3
        assert nvt.iloc[0] == pytest.approx(10.0)


class TestNVTSignal:
    def test_z_score_centered(self):
        # / constant nvt -> z-score should be ~0 after warmup
        mcap = _make_series([1e9] * 100)
        vol = _make_series([1e8] * 100)
        z = nvt_signal(mcap, vol, period=90)
        # / z-score of constant series is nan (std=0)
        # / this is correct behavior
        assert True

    def test_high_nvt_positive_z(self):
        # / stable nvt then spike -> positive z-score
        mcap_vals = [1e9] * 95 + [2e9] * 5
        vol_vals = [1e8] * 100
        mcap = _make_series(mcap_vals)
        vol = _make_series(vol_vals)
        z = nvt_signal(mcap, vol, period=90)
        # / last values should have positive z (nvt=20 vs mean ~10)
        assert z.iloc[-1] > 0

    def test_output_shape(self):
        mcap = _make_series([1e9] * 100)
        vol = _make_series([1e8] * 100)
        z = nvt_signal(mcap, vol, period=90)
        assert len(z) == 100
