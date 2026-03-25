# / tests for sec filings: insider trades fetch, parse, store, rate limiting

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.sec_filings import (
    _get_transactions,
    _get_user_agent,
    _safe_get,
    fetch_all_insider_trades,
    fetch_insider_trades,
    log_data_quality_issue,
    store_insider_trades,
)


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


class TestSafeGet:
    def test_gets_existing_attr(self):
        obj = MagicMock()
        obj.name = "test"
        assert _safe_get(obj, "name") == "test"

    def test_returns_default_for_missing_attr(self):
        obj = MagicMock(spec=[])
        assert _safe_get(obj, "missing", "default") == "default"

    def test_returns_default_for_none_value(self):
        obj = MagicMock()
        obj.name = None
        assert _safe_get(obj, "name", "fallback") == "fallback"

    def test_returns_default_on_exception(self):
        obj = MagicMock()
        type(obj).name = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        assert _safe_get(obj, "name", "safe") == "safe"


class TestGetUserAgent:
    def test_returns_env_var_when_set(self):
        with patch.dict("os.environ", {"SEC_EDGAR_USER_AGENT": "MyAgent my@email.com"}):
            assert _get_user_agent() == "MyAgent my@email.com"

    def test_returns_default_when_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _get_user_agent()
            assert "QuantTrader" in result


class TestGetTransactions:
    def test_extracts_buy_transaction(self):
        item = MagicMock()
        item.transaction_code = "P"
        item.shares = 1000
        item.price_per_share = 150.0

        form4 = MagicMock()
        form4.non_derivative_transactions = [item]
        form4.transactions = None
        form4.derivative_transactions = None

        txns = _get_transactions(form4)
        assert len(txns) == 1
        assert txns[0]["type"] == "buy"
        assert txns[0]["shares"] == 1000

    def test_extracts_sell_transaction(self):
        item = MagicMock()
        item.transaction_code = "S"
        item.shares = 500
        item.price_per_share = 200.0

        form4 = MagicMock()
        form4.non_derivative_transactions = [item]
        form4.transactions = None
        form4.derivative_transactions = None

        txns = _get_transactions(form4)
        assert len(txns) == 1
        assert txns[0]["type"] == "sell"

    def test_extracts_option_exercise(self):
        item = MagicMock()
        item.transaction_code = "M"
        item.shares = 2000
        item.price_per_share = 50.0

        form4 = MagicMock()
        form4.non_derivative_transactions = None
        form4.transactions = [item]
        form4.derivative_transactions = None

        txns = _get_transactions(form4)
        assert len(txns) == 1
        assert txns[0]["type"] == "option_exercise"

    def test_handles_unknown_code(self):
        item = MagicMock()
        item.transaction_code = "X"
        item.shares = 100
        item.price_per_share = 10.0

        form4 = MagicMock()
        form4.non_derivative_transactions = [item]
        form4.transactions = None
        form4.derivative_transactions = None

        txns = _get_transactions(form4)
        assert txns[0]["type"] == "X"

    def test_handles_no_transactions(self):
        form4 = MagicMock()
        form4.non_derivative_transactions = None
        form4.transactions = None
        form4.derivative_transactions = None

        txns = _get_transactions(form4)
        assert txns == []

    def test_continues_on_iteration_error(self):
        form4 = MagicMock()
        form4.non_derivative_transactions = MagicMock(side_effect=TypeError("not iterable"))
        form4.transactions = None
        form4.derivative_transactions = None

        # / should not crash
        txns = _get_transactions(form4)
        assert txns == []


class TestFetchInsiderTrades:
    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        with patch("src.data.sec_filings._fetch_insider_trades_sync", side_effect=Exception("fail")):
            with patch("src.data.sec_filings._edgar_delay", 0):
                trades = await fetch_insider_trades("AAPL", days=90)
                assert trades == []

    @pytest.mark.asyncio
    async def test_returns_trades_on_success(self):
        mock_trades = [
            {"symbol": "AAPL", "filing_date": date.today(),
             "insider_name": "Tim Cook", "insider_title": "CEO",
             "transaction_type": "sell", "shares": Decimal("50000"),
             "price_per_share": Decimal("150"), "total_value": Decimal("7500000")},
        ]

        with patch("src.data.sec_filings._fetch_insider_trades_sync", return_value=mock_trades):
            with patch("src.data.sec_filings._edgar_delay", 0):
                trades = await fetch_insider_trades("AAPL", days=90)
                assert len(trades) == 1
                assert trades[0]["insider_name"] == "Tim Cook"


class TestFetchAllInsiderTrades:
    @pytest.mark.asyncio
    async def test_fetches_all_symbols(self):
        async def mock_fetch(sym, days=90):
            return [{"symbol": sym, "filing_date": date.today(),
                     "insider_name": "Insider", "insider_title": "",
                     "transaction_type": "buy", "shares": Decimal("100"),
                     "price_per_share": Decimal("50"), "total_value": Decimal("5000")}]

        with patch("src.data.sec_filings.fetch_insider_trades", side_effect=mock_fetch):
            trades = await fetch_all_insider_trades(["AAPL", "MSFT"], days=90)
            assert len(trades) == 2

    @pytest.mark.asyncio
    async def test_continues_when_symbol_returns_empty(self):
        async def mock_fetch(sym, days=90):
            if sym == "FAKE":
                return []
            return [{"symbol": sym, "filing_date": date.today(),
                     "insider_name": "Insider", "insider_title": "",
                     "transaction_type": "buy", "shares": Decimal("100"),
                     "price_per_share": Decimal("50"), "total_value": Decimal("5000")}]

        with patch("src.data.sec_filings.fetch_insider_trades", side_effect=mock_fetch):
            trades = await fetch_all_insider_trades(["AAPL", "FAKE", "MSFT"], days=90)
            assert len(trades) == 2


class TestStoreInsiderTrades:
    @pytest.mark.asyncio
    async def test_stores_trades(self):
        trades = [
            {"symbol": "AAPL", "filing_date": date.today(),
             "insider_name": "Tim Cook", "insider_title": "CEO",
             "transaction_type": "sell", "shares": Decimal("50000"),
             "price_per_share": Decimal("150"), "total_value": Decimal("7500000")},
        ]

        mock_conn = AsyncMock()
        pool = _mock_pool(mock_conn)

        count = await store_insider_trades(pool, trades)
        assert count == 1
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_trades_returns_zero(self):
        mock_pool = AsyncMock()
        count = await store_insider_trades(mock_pool, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_continues_on_insert_error(self):
        trades = [
            {"symbol": "AAPL", "filing_date": date.today(),
             "insider_name": "A", "insider_title": "",
             "transaction_type": "buy", "shares": Decimal("100"),
             "price_per_share": Decimal("50"), "total_value": Decimal("5000")},
            {"symbol": "MSFT", "filing_date": date.today(),
             "insider_name": "B", "insider_title": "",
             "transaction_type": "sell", "shares": Decimal("200"),
             "price_per_share": Decimal("300"), "total_value": Decimal("60000")},
        ]

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = [Exception("duplicate"), None]
        pool = _mock_pool(mock_conn)

        count = await store_insider_trades(pool, trades)
        # / first fails, second succeeds
        assert count == 1


class TestLogDataQualityIssue:
    @pytest.mark.asyncio
    async def test_logs_issue(self):
        mock_conn = AsyncMock()
        pool = _mock_pool(mock_conn)

        await log_data_quality_issue(pool, "AAPL", "parse error on filing X")
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_crash_on_db_error(self):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.side_effect = Exception("db down")
        pool = MagicMock()
        pool.acquire.return_value = mock_ctx

        # / should not raise
        await log_data_quality_issue(pool, "AAPL", "some error")
