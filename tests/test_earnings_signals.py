# / tests for earnings signals analysis

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.earnings_signals import (
    EarningsSignal,
    analyze_earnings,
    compute_earnings_signal,
    fetch_earnings,
)


class TestComputeEarningsSignal:
    def test_strong_beats(self):
        data = {
            "symbol": "AAPL",
            "quarters": [
                {"period": "Q1", "actual": 1.50, "estimate": 1.20, "surprise_pct": 0.25},
                {"period": "Q2", "actual": 1.40, "estimate": 1.10, "surprise_pct": 0.273},
                {"period": "Q3", "actual": 1.30, "estimate": 1.05, "surprise_pct": 0.238},
                {"period": "Q4", "actual": 1.20, "estimate": 1.00, "surprise_pct": 0.20},
            ],
        }
        result = compute_earnings_signal(data)
        assert result.signal == "bullish"
        assert result.strength > 50
        assert result.consecutive_beats >= 4
        assert result.surprise_pct == 0.25

    def test_strong_misses(self):
        data = {
            "symbol": "BAD",
            "quarters": [
                {"period": "Q1", "actual": 0.80, "estimate": 1.20, "surprise_pct": -0.333},
                {"period": "Q2", "actual": 0.90, "estimate": 1.10, "surprise_pct": -0.182},
                {"period": "Q3", "actual": 0.85, "estimate": 1.05, "surprise_pct": -0.190},
            ],
        }
        result = compute_earnings_signal(data)
        assert result.signal == "bearish"
        assert result.strength > 50
        assert result.consecutive_beats <= -3

    def test_mixed_results(self):
        data = {
            "symbol": "MIX",
            "quarters": [
                {"period": "Q1", "actual": 1.02, "estimate": 1.00, "surprise_pct": 0.02},
                {"period": "Q2", "actual": 0.98, "estimate": 1.00, "surprise_pct": -0.02},
            ],
        }
        result = compute_earnings_signal(data)
        # / small surprises within threshold -> neutral
        assert result.signal == "neutral"
        assert result.strength < 20

    def test_no_data(self):
        data = {"symbol": "EMPTY", "quarters": []}
        result = compute_earnings_signal(data)
        assert result.signal == "neutral"
        assert result.strength == 0.0
        assert result.surprise_pct is None

    def test_no_surprise_data(self):
        data = {
            "symbol": "PARTIAL",
            "quarters": [
                {"period": "Q1", "actual": 1.0, "estimate": None, "surprise_pct": None},
            ],
        }
        result = compute_earnings_signal(data)
        assert result.signal == "neutral"

    def test_single_big_beat(self):
        data = {
            "symbol": "BEAT",
            "quarters": [
                {"period": "Q1", "actual": 2.00, "estimate": 1.00, "surprise_pct": 1.0},
            ],
        }
        result = compute_earnings_signal(data)
        assert result.signal == "bullish"
        assert result.surprise_pct == 1.0

    def test_avg_surprise_calculated(self):
        data = {
            "symbol": "AVG",
            "quarters": [
                {"period": "Q1", "actual": 1.10, "estimate": 1.00, "surprise_pct": 0.10},
                {"period": "Q2", "actual": 1.20, "estimate": 1.00, "surprise_pct": 0.20},
            ],
        }
        result = compute_earnings_signal(data)
        assert result.avg_surprise_4q == 0.15


class TestFetchEarnings:
    @pytest.mark.asyncio
    async def test_returns_data(self):
        import pandas as pd

        mock_earnings = pd.DataFrame(
            {"Actual": [1.5, 1.3], "Estimate": [1.2, 1.1]},
            index=["Q1 2026", "Q4 2025"],
        )
        mock_ticker = MagicMock()
        mock_ticker.quarterly_earnings = mock_earnings
        mock_ticker.earnings_dates = None
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = await fetch_earnings("AAPL")
            assert result is not None
            assert result["symbol"] == "AAPL"
            assert len(result["quarters"]) == 2

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = await fetch_earnings("FAIL")
            assert result is None


class TestAnalyzeEarnings:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        earnings_data = {
            "symbol": "AAPL",
            "quarters": [
                {"period": "Q1", "actual": 1.5, "estimate": 1.2, "surprise_pct": 0.25},
            ],
        }

        with patch("src.analysis.earnings_signals.fetch_earnings", return_value=earnings_data):
            result = await analyze_earnings("AAPL")
            assert result is not None
            assert result.symbol == "AAPL"
            assert result.signal == "bullish"

    @pytest.mark.asyncio
    async def test_returns_none_on_fetch_failure(self):
        with patch("src.analysis.earnings_signals.fetch_earnings", return_value=None):
            result = await analyze_earnings("FAIL")
            assert result is None
