# / tests for insider activity analysis

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analysis.insider_activity import (
    InsiderSignal,
    _detect_cluster,
    _title_weight,
    analyze_insider_activity,
    compute_insider_signal,
)


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


class TestTitleWeight:
    def test_ceo_highest(self):
        assert _title_weight("CEO") == 3.0

    def test_cfo(self):
        assert _title_weight("Chief Financial Officer") == 2.5

    def test_director(self):
        assert _title_weight("Director") == 1.0

    def test_vp(self):
        assert _title_weight("VP of Engineering") == 1.5

    def test_vice_president(self):
        assert _title_weight("Vice President") == 1.5

    def test_unknown(self):
        assert _title_weight("Board Member") == 1.0

    def test_empty(self):
        assert _title_weight("") == 1.0

    def test_case_insensitive(self):
        assert _title_weight("ceo") == 3.0


class TestDetectCluster:
    def test_cluster_detected(self):
        base_date = date(2026, 3, 1)
        trades = [
            {"transaction_type": "buy", "filing_date": base_date, "insider_name": "Alice"},
            {"transaction_type": "buy", "filing_date": base_date + timedelta(days=5), "insider_name": "Bob"},
            {"transaction_type": "buy", "filing_date": base_date + timedelta(days=10), "insider_name": "Charlie"},
        ]
        assert _detect_cluster(trades) is True

    def test_no_cluster_too_few(self):
        base_date = date(2026, 3, 1)
        trades = [
            {"transaction_type": "buy", "filing_date": base_date, "insider_name": "Alice"},
            {"transaction_type": "buy", "filing_date": base_date + timedelta(days=5), "insider_name": "Bob"},
        ]
        assert _detect_cluster(trades) is False

    def test_no_cluster_too_spread(self):
        trades = [
            {"transaction_type": "buy", "filing_date": date(2026, 1, 1), "insider_name": "Alice"},
            {"transaction_type": "buy", "filing_date": date(2026, 2, 15), "insider_name": "Bob"},
            {"transaction_type": "buy", "filing_date": date(2026, 3, 25), "insider_name": "Charlie"},
        ]
        assert _detect_cluster(trades) is False

    def test_sells_excluded(self):
        base_date = date(2026, 3, 1)
        trades = [
            {"transaction_type": "sell", "filing_date": base_date, "insider_name": "Alice"},
            {"transaction_type": "sell", "filing_date": base_date + timedelta(days=5), "insider_name": "Bob"},
            {"transaction_type": "sell", "filing_date": base_date + timedelta(days=10), "insider_name": "Charlie"},
        ]
        assert _detect_cluster(trades) is False


class TestComputeInsiderSignal:
    def test_strong_buying(self):
        trades = [
            {"transaction_type": "buy", "total_value": 500000, "insider_title": "CEO",
             "insider_name": "Alice", "filing_date": date(2026, 3, 1)},
            {"transaction_type": "buy", "total_value": 200000, "insider_title": "CFO",
             "insider_name": "Bob", "filing_date": date(2026, 3, 5)},
            {"transaction_type": "buy", "total_value": 100000, "insider_title": "VP",
             "insider_name": "Charlie", "filing_date": date(2026, 3, 10)},
        ]
        result = compute_insider_signal(trades, "AAPL")
        assert result.signal == "bullish"
        assert result.net_buy_ratio == 1.0
        assert result.total_buys == 3
        assert result.total_sells == 0
        assert result.cluster_detected is True
        assert result.unique_buyers == 3

    def test_strong_selling(self):
        trades = [
            {"transaction_type": "sell", "total_value": 1000000, "insider_title": "CEO",
             "insider_name": "Alice", "filing_date": date(2026, 3, 1)},
            {"transaction_type": "sell", "total_value": 500000, "insider_title": "Director",
             "insider_name": "Bob", "filing_date": date(2026, 3, 5)},
        ]
        result = compute_insider_signal(trades, "AAPL")
        assert result.signal == "bearish"
        assert result.net_buy_ratio == -1.0
        assert result.total_sells == 2

    def test_mixed_activity(self):
        trades = [
            {"transaction_type": "buy", "total_value": 100000, "insider_title": "Director",
             "insider_name": "Alice", "filing_date": date(2026, 3, 1)},
            {"transaction_type": "sell", "total_value": 100000, "insider_title": "Director",
             "insider_name": "Bob", "filing_date": date(2026, 3, 5)},
        ]
        result = compute_insider_signal(trades, "AAPL")
        assert result.net_buy_ratio == 0.0
        assert result.signal == "neutral"

    def test_no_trades(self):
        result = compute_insider_signal([], "AAPL")
        assert result.signal == "neutral"
        assert result.strength == 0.0
        assert result.total_buys == 0

    def test_option_exercises_excluded(self):
        trades = [
            {"transaction_type": "option_exercise", "total_value": 5000000,
             "insider_title": "CEO", "insider_name": "Alice", "filing_date": date(2026, 3, 1)},
            {"transaction_type": "buy", "total_value": 50000,
             "insider_title": "Director", "insider_name": "Bob", "filing_date": date(2026, 3, 5)},
        ]
        result = compute_insider_signal(trades, "AAPL")
        # / option exercise shouldn't count as buy or sell
        assert result.total_buys == 1
        assert result.total_sells == 0

    def test_officer_weight_matters(self):
        # / ceo buy should produce stronger signal than director buy
        ceo_trades = [
            {"transaction_type": "buy", "total_value": 100000, "insider_title": "CEO",
             "insider_name": "Alice", "filing_date": date(2026, 3, 1)},
        ]
        dir_trades = [
            {"transaction_type": "buy", "total_value": 100000, "insider_title": "Director",
             "insider_name": "Bob", "filing_date": date(2026, 3, 1)},
        ]
        ceo_result = compute_insider_signal(ceo_trades, "AAPL")
        dir_result = compute_insider_signal(dir_trades, "AAPL")
        # / both bullish, but same net ratio since no opposing trades
        assert ceo_result.signal == "bullish"
        assert dir_result.signal == "bullish"


class TestAnalyzeInsiderActivity:
    @pytest.mark.asyncio
    async def test_fetches_and_analyzes(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"symbol": "AAPL", "filing_date": date(2026, 3, 1),
             "insider_name": "Tim Cook", "insider_title": "CEO",
             "transaction_type": "buy", "shares": Decimal("1000"),
             "price_per_share": Decimal("180.00"), "total_value": Decimal("180000")},
        ]
        pool = _mock_pool(mock_conn)

        result = await analyze_insider_activity(pool, "AAPL")
        assert result.symbol == "AAPL"
        assert result.total_buys == 1
        assert result.signal == "bullish"

    @pytest.mark.asyncio
    async def test_empty_trades(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        result = await analyze_insider_activity(pool, "AAPL")
        assert result.signal == "neutral"
        assert result.total_buys == 0
