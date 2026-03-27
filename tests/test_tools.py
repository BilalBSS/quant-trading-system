# / tests for agent pipeline db helpers

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.tools import (
    _STATUS_TABLES,
    analysis_data_to_dict,
    dict_to_analysis_data,
    fetch_analysis_score,
    fetch_daily_synthesis,
    fetch_pending_signals,
    fetch_pending_trades,
    fetch_recent_trades,
    fetch_strategy_scores,
    store_analysis_score,
    store_approved_trade,
    store_daily_synthesis,
    store_evolution_log,
    store_strategy_score,
    store_trade_log,
    store_trade_signal,
    update_trade_status,
)
from src.strategies.base_strategy import AnalysisData


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


# -- store_analysis_score --

class TestStoreAnalysisScore:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 42}
        pool = _mock_pool(mock_conn)

        result = await store_analysis_score(
            pool, "AAPL", date(2026, 3, 25), 85.0, 72.0, 78.5,
            "bull", 0.85, True, {"notes": "strong"},
        )
        assert result == 42

    @pytest.mark.asyncio
    async def test_passes_correct_args(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_analysis_score(
            pool, "MSFT", date(2026, 1, 1), 90.0, None, 90.0,
            "sideways", 0.6, True,
        )
        args = mock_conn.fetchrow.call_args[0]
        # / positional: sql, symbol, date, fund, tech, comp, regime, conf, used_fund, details
        assert args[1] == "MSFT"
        assert args[2] == date(2026, 1, 1)
        assert args[3] == Decimal("90.0")
        assert args[4] is None  # / technical_score was None
        assert args[5] == Decimal("90.0")
        assert args[6] == "sideways"
        assert args[7] == Decimal("0.6")
        assert args[8] is True

    @pytest.mark.asyncio
    async def test_decimal_conversion(self):
        # / floats become Decimal for db storage
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_analysis_score(
            pool, "GOOG", date(2026, 1, 1), 55.5, 60.3, 57.9,
            "bear", 0.72, False,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert isinstance(args[3], Decimal)
        assert isinstance(args[4], Decimal)
        assert isinstance(args[5], Decimal)
        assert isinstance(args[7], Decimal)

    @pytest.mark.asyncio
    async def test_json_dumps_details(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        details = {"ratio_scores": [1, 2, 3]}
        await store_analysis_score(
            pool, "TSLA", date(2026, 1, 1), 50.0, 50.0, 50.0,
            None, None, True, details,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[9] == json.dumps(details)

    @pytest.mark.asyncio
    async def test_none_details_passes_none(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_analysis_score(
            pool, "TSLA", date(2026, 1, 1), 50.0, 50.0, 50.0,
            None, None, True, None,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[9] is None

    @pytest.mark.asyncio
    async def test_sql_contains_upsert(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_analysis_score(
            pool, "X", date(2026, 1, 1), 1.0, 1.0, 1.0,
            None, None, False,
        )
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "RETURNING id" in sql


# -- fetch_analysis_score --

class TestFetchAnalysisScore:
    @pytest.mark.asyncio
    async def test_returns_dict(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1, "symbol": "AAPL", "composite_score": 80}
        pool = _mock_pool(mock_conn)

        result = await fetch_analysis_score(pool, "AAPL", date(2026, 3, 25))
        assert result == {"id": 1, "symbol": "AAPL", "composite_score": 80}

    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        result = await fetch_analysis_score(pool, "FAKE")
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_symbol_and_date(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        await fetch_analysis_score(pool, "NVDA", date(2026, 6, 1))
        args = mock_conn.fetchrow.call_args[0]
        assert args[1] == "NVDA"
        assert args[2] == date(2026, 6, 1)


# -- store_trade_signal --

class TestStoreTradeSignal:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 7}
        pool = _mock_pool(mock_conn)

        result = await store_trade_signal(
            pool, "strategy_001", "AAPL", "buy", 0.85, "bull",
        )
        assert result == 7

    @pytest.mark.asyncio
    async def test_strength_as_decimal(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_trade_signal(pool, "s1", "MSFT", "sell", 0.42, None)
        args = mock_conn.fetchrow.call_args[0]
        assert args[4] == Decimal("0.42")

    @pytest.mark.asyncio
    async def test_details_json_dumped(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        details = {"reason": "oversold"}
        await store_trade_signal(pool, "s1", "X", "buy", 0.5, None, details)
        args = mock_conn.fetchrow.call_args[0]
        assert args[6] == json.dumps(details)

    @pytest.mark.asyncio
    async def test_sql_has_pending_status(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_trade_signal(pool, "s1", "X", "buy", 0.5, None)
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "'pending'" in sql


# -- fetch_pending_signals --

class TestFetchPendingSignals:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"id": 1, "symbol": "AAPL"},
            {"id": 2, "symbol": "MSFT"},
        ]
        pool = _mock_pool(mock_conn)

        result = await fetch_pending_signals(pool)
        assert len(result) == 2
        assert result[0] == {"id": 1, "symbol": "AAPL"}

    @pytest.mark.asyncio
    async def test_returns_empty_list(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        result = await fetch_pending_signals(pool)
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_limit(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        await fetch_pending_signals(pool, limit=10)
        args = mock_conn.fetch.call_args[0]
        assert args[1] == 10


# -- store_approved_trade --

class TestStoreApprovedTrade:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 99}
        pool = _mock_pool(mock_conn)

        result = await store_approved_trade(pool, 7, "AAPL", "buy", 10.0)
        assert result == 99

    @pytest.mark.asyncio
    async def test_qty_as_decimal(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_approved_trade(pool, 1, "MSFT", "sell", 5.5, "limit", "s1")
        args = mock_conn.fetchrow.call_args[0]
        assert args[4] == Decimal("5.5")

    @pytest.mark.asyncio
    async def test_default_order_type(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_approved_trade(pool, 1, "X", "buy", 1.0)
        args = mock_conn.fetchrow.call_args[0]
        assert args[5] == "market"


# -- fetch_pending_trades --

class TestFetchPendingTrades:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1, "symbol": "GOOG"}]
        pool = _mock_pool(mock_conn)

        result = await fetch_pending_trades(pool)
        assert result == [{"id": 1, "symbol": "GOOG"}]

    @pytest.mark.asyncio
    async def test_empty_result(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        result = await fetch_pending_trades(pool)
        assert result == []


# -- update_trade_status --

class TestUpdateTradeStatus:
    @pytest.mark.asyncio
    async def test_valid_table_trade_signals(self):
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 1"
        pool = _mock_pool(mock_conn)

        result = await update_trade_status(pool, "trade_signals", 5, "approved")
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_table_approved_trades(self):
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 1"
        pool = _mock_pool(mock_conn)

        result = await update_trade_status(pool, "approved_trades", 3, "executed")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_row_updated(self):
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 0"
        pool = _mock_pool(mock_conn)

        result = await update_trade_status(pool, "trade_signals", 999, "rejected")
        assert result is False

    @pytest.mark.asyncio
    async def test_rejects_invalid_table(self):
        pool = MagicMock()
        with pytest.raises(ValueError, match="invalid table"):
            await update_trade_status(pool, "users; DROP TABLE trade_signals;--", 1, "x")

    @pytest.mark.asyncio
    async def test_rejects_trade_log_table(self):
        pool = MagicMock()
        with pytest.raises(ValueError):
            await update_trade_status(pool, "trade_log", 1, "x")

    def test_status_tables_whitelist(self):
        assert _STATUS_TABLES == {"trade_signals", "approved_trades"}


# -- store_trade_log --

class TestStoreTradeLog:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 200}
        pool = _mock_pool(mock_conn)

        result = await store_trade_log(
            pool, 99, "AAPL", "buy", 10.0, 150.50, "ord_123", "alpaca", "bull", 25.0,
        )
        assert result == 200

    @pytest.mark.asyncio
    async def test_decimal_conversions(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_trade_log(
            pool, 1, "MSFT", "sell", 5.0, 300.25, None, None, None, -10.5,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[4] == Decimal("5.0")
        assert args[5] == Decimal("300.25")
        assert args[9] == Decimal("-10.5")

    @pytest.mark.asyncio
    async def test_none_pnl_stays_none(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_trade_log(
            pool, None, "X", "buy", 1.0, 10.0, None, None, None, None,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[9] is None

    @pytest.mark.asyncio
    async def test_details_json_dumped(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        details = {"slippage": 0.01}
        await store_trade_log(
            pool, 1, "X", "buy", 1.0, 10.0, None, None, None, None,
            details=details,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[11] == json.dumps(details)


# -- store_strategy_score --

class TestStoreStrategyScore:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 5}
        pool = _mock_pool(mock_conn)

        result = await store_strategy_score(
            pool, "strat_001", date(2026, 1, 1), date(2026, 3, 1),
            1.42, -0.12, 0.58, 0.18, 50,
        )
        assert result == 5

    @pytest.mark.asyncio
    async def test_decimal_conversions(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_strategy_score(
            pool, "s1", date(2026, 1, 1), date(2026, 3, 1),
            1.5, -0.08, 0.65, 0.15, 100,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[4] == Decimal("1.5")
        assert args[5] == Decimal("-0.08")
        assert args[6] == Decimal("0.65")
        assert args[7] == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_none_brier_stays_none(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        await store_strategy_score(
            pool, "s1", date(2026, 1, 1), date(2026, 3, 1),
            1.0, -0.05, 0.50, None, 10,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[7] is None

    @pytest.mark.asyncio
    async def test_regime_breakdown_json(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        breakdown = {"bull": 1.5, "bear": -0.3}
        await store_strategy_score(
            pool, "s1", date(2026, 1, 1), date(2026, 3, 1),
            1.0, -0.05, 0.50, None, 10, breakdown,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[9] == json.dumps(breakdown)


# -- fetch_strategy_scores --

class TestFetchStrategyScores:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"id": 1, "strategy_id": "s1"},
            {"id": 2, "strategy_id": "s2"},
        ]
        pool = _mock_pool(mock_conn)

        result = await fetch_strategy_scores(pool)
        assert len(result) == 2
        assert result[0]["strategy_id"] == "s1"

    @pytest.mark.asyncio
    async def test_empty_result(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        result = await fetch_strategy_scores(pool)
        assert result == []


# -- store_evolution_log --

class TestStoreEvolutionLog:
    @pytest.mark.asyncio
    async def test_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 10}
        pool = _mock_pool(mock_conn)

        result = await store_evolution_log(
            pool, 3, "kill", "strat_005", None, "underperforming",
        )
        assert result == 10

    @pytest.mark.asyncio
    async def test_details_json_dumped(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        details = {"sharpe": 0.3}
        await store_evolution_log(
            pool, 1, "mutate", "s2", "s1", "low sharpe", details,
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[6] == json.dumps(details)


# -- fetch_recent_trades --

class TestFetchRecentTrades:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [{"id": 1}]
        pool = _mock_pool(mock_conn)

        result = await fetch_recent_trades(pool)
        assert result == [{"id": 1}]

    @pytest.mark.asyncio
    async def test_filters_by_strategy(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        await fetch_recent_trades(pool, strategy_id="s1", limit=25)
        args = mock_conn.fetch.call_args[0]
        assert "strategy_id" in args[0]
        assert args[1] == "s1"
        assert args[2] == 25

    @pytest.mark.asyncio
    async def test_no_strategy_filter(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        await fetch_recent_trades(pool, limit=10)
        args = mock_conn.fetch.call_args[0]
        assert "strategy_id" not in args[0]
        assert args[1] == 10

    @pytest.mark.asyncio
    async def test_empty_result(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        pool = _mock_pool(mock_conn)

        result = await fetch_recent_trades(pool, strategy_id="nonexistent")
        assert result == []


# -- analysis_data_to_dict / dict_to_analysis_data --

class TestAnalysisDataRoundTrip:
    def test_full_roundtrip(self):
        data = AnalysisData(
            pe_ratio=25.0,
            pe_forward=22.0,
            ps_ratio=8.0,
            peg_ratio=1.5,
            revenue_growth=0.12,
            fcf_margin=0.20,
            debt_to_equity=0.8,
            sector_pe_avg=30.0,
            sector_ps_avg=10.0,
            dcf_upside=0.15,
            insider_net_buy_ratio=0.6,
            earnings_surprise_pct=5.0,
            consecutive_beats=3,
            fundamental_score=75.0,
        )
        d = analysis_data_to_dict(data)
        restored = dict_to_analysis_data(d)
        assert restored == data

    def test_all_none_roundtrip(self):
        data = AnalysisData()
        d = analysis_data_to_dict(data)
        restored = dict_to_analysis_data(d)
        assert restored == data
        assert restored.consecutive_beats == 0

    def test_partial_data_roundtrip(self):
        data = AnalysisData(pe_ratio=20.0, fundamental_score=60.0)
        d = analysis_data_to_dict(data)
        restored = dict_to_analysis_data(d)
        assert restored.pe_ratio == 20.0
        assert restored.fundamental_score == 60.0
        assert restored.ps_ratio is None

    def test_dict_has_all_keys(self):
        data = AnalysisData()
        d = analysis_data_to_dict(data)
        expected_keys = {
            "pe_ratio", "pe_forward", "ps_ratio", "peg_ratio",
            "revenue_growth", "fcf_margin", "debt_to_equity",
            "sector_pe_avg", "sector_ps_avg", "dcf_upside",
            "insider_net_buy_ratio", "earnings_surprise_pct",
            "consecutive_beats", "fundamental_score",
            "nvt_ratio", "funding_rate", "exchange_flow_ratio",
            "news_sentiment_score", "ai_consensus",
        }
        assert set(d.keys()) == expected_keys

    def test_consecutive_beats_default(self):
        # / missing key defaults to 0
        d = {"pe_ratio": 10.0}
        result = dict_to_analysis_data(d)
        assert result.consecutive_beats == 0

    def test_extra_keys_ignored(self):
        # / dict_to_analysis_data should not crash on extra keys
        d = {"pe_ratio": 10.0, "unknown_field": "ignored"}
        result = dict_to_analysis_data(d)
        assert result.pe_ratio == 10.0


# -- store_daily_synthesis / fetch_daily_synthesis --

class TestDailySynthesis:
    @pytest.mark.asyncio
    async def test_store_returns_id(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 5}
        pool = _mock_pool(mock_conn)

        result = await store_daily_synthesis(
            pool, date(2026, 3, 27), "deepseek-reasoner",
            [{"symbol": "NVDA"}], [{"symbol": "MRNA"}],
            "moderate risk", {"AAPL": "watching"}, "raw text",
        )
        assert result == 5

    @pytest.mark.asyncio
    async def test_store_passes_json_args(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 1}
        pool = _mock_pool(mock_conn)

        buys = [{"symbol": "CRM", "score": 53.0}]
        avoids = [{"symbol": "TSLA", "score": 3.3}]
        notes = {"NVDA": "strong momentum"}

        await store_daily_synthesis(
            pool, date(2026, 3, 27), "deepseek-reasoner",
            buys, avoids, "low risk", notes, "raw",
        )
        args = mock_conn.fetchrow.call_args[0]
        assert args[1] == date(2026, 3, 27)
        assert args[2] == "deepseek-reasoner"
        # / top_buys and top_avoids are json-serialized
        assert json.loads(args[3]) == buys
        assert json.loads(args[4]) == avoids

    @pytest.mark.asyncio
    async def test_fetch_latest(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "id": 1, "date": date(2026, 3, 27),
            "model": "deepseek-reasoner", "top_buys": "[]",
        }
        pool = _mock_pool(mock_conn)

        result = await fetch_daily_synthesis(pool)
        assert result is not None
        assert result["model"] == "deepseek-reasoner"

    @pytest.mark.asyncio
    async def test_fetch_by_date(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"id": 2, "date": date(2026, 3, 26)}
        pool = _mock_pool(mock_conn)

        result = await fetch_daily_synthesis(pool, target_date=date(2026, 3, 26))
        assert result["date"] == date(2026, 3, 26)
        args = mock_conn.fetchrow.call_args[0]
        assert args[1] == date(2026, 3, 26)

    @pytest.mark.asyncio
    async def test_fetch_returns_none_when_empty(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        pool = _mock_pool(mock_conn)

        result = await fetch_daily_synthesis(pool)
        assert result is None
