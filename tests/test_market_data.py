# / tests for market data fetching, parsing, validation, storage, backfill

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.market_data import (
    _parse_bar,
    fetch_bars,
    fetch_bars_alpaca,
    fetch_bars_yfinance,
    fetch_latest_quote,
    store_bars,
    backfill,
)


def _mock_pool(mock_conn):
    # / helper to create a properly mocked asyncpg pool
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    mock_pool = MagicMock()
    mock_pool.acquire.return_value = mock_ctx
    return mock_pool


class TestParseBar:
    def test_parses_alpaca_bar(self):
        bar = {
            "t": "2024-01-15T05:00:00Z",
            "o": 150.0,
            "h": 155.0,
            "l": 149.0,
            "c": 153.5,
            "v": 1000000,
            "vw": 152.3,
        }
        result = _parse_bar("AAPL", bar)
        assert result["symbol"] == "AAPL"
        assert result["date"] == date(2024, 1, 15)
        assert result["open"] == Decimal("150.0")
        assert result["high"] == Decimal("155.0")
        assert result["low"] == Decimal("149.0")
        assert result["close"] == Decimal("153.5")
        assert result["volume"] == 1000000
        assert result["vwap"] == Decimal("152.3")

    def test_parses_bar_without_vwap(self):
        bar = {"t": "2024-01-15T05:00:00Z", "o": 100, "h": 110, "l": 90, "c": 105, "v": 500}
        result = _parse_bar("SPY", bar)
        assert result["vwap"] is None

    def test_parses_bar_without_volume(self):
        bar = {"t": "2024-01-15T05:00:00Z", "o": 100, "h": 110, "l": 90, "c": 105}
        result = _parse_bar("SPY", bar)
        assert result["volume"] == 0

    def test_returns_none_for_unparseable_timestamp(self):
        bar = {"t": "not-a-date", "o": 100, "h": 110, "l": 90, "c": 105, "v": 500}
        result = _parse_bar("SPY", bar)
        assert result is None


class TestFetchBarsAlpaca:
    @pytest.mark.asyncio
    async def test_fetches_equity_bars(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "bars": [
                {"t": "2024-01-15T05:00:00Z", "o": 100, "h": 110, "l": 90, "c": 105, "v": 1000, "vw": 102},
            ],
            "next_page_token": None,
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_data.httpx.AsyncClient", return_value=mock_client):
            with patch("src.data.market_data._rate_delay", 0):
                bars = await fetch_bars_alpaca("AAPL", date(2024, 1, 1), date(2024, 1, 31))

        assert len(bars) == 1
        assert bars[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_fetches_crypto_bars(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "bars": {
                "BTC/USD": [
                    {"t": "2024-01-15T05:00:00Z", "o": 42000, "h": 43000, "l": 41000, "c": 42500, "v": 100},
                ]
            },
            "next_page_token": None,
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_data.httpx.AsyncClient", return_value=mock_client):
            with patch("src.data.market_data._rate_delay", 0):
                bars = await fetch_bars_alpaca("BTC-USD", date(2024, 1, 1), date(2024, 1, 31))

        assert len(bars) == 1
        assert bars[0]["symbol"] == "BTC-USD"

    @pytest.mark.asyncio
    async def test_paginates(self):
        page1 = MagicMock()
        page1.raise_for_status = MagicMock()
        page1.json.return_value = {
            "bars": [{"t": "2024-01-15T05:00:00Z", "o": 100, "h": 110, "l": 90, "c": 105, "v": 1000, "vw": 102}],
            "next_page_token": "token123",
        }
        page2 = MagicMock()
        page2.raise_for_status = MagicMock()
        page2.json.return_value = {
            "bars": [{"t": "2024-01-16T05:00:00Z", "o": 105, "h": 115, "l": 100, "c": 110, "v": 2000, "vw": 108}],
            "next_page_token": None,
        }

        mock_client = AsyncMock()
        mock_client.get.side_effect = [page1, page2]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_data.httpx.AsyncClient", return_value=mock_client):
            with patch("src.data.market_data._rate_delay", 0):
                bars = await fetch_bars_alpaca("AAPL", date(2024, 1, 1), date(2024, 1, 31))

        assert len(bars) == 2


class TestFetchBarsYfinance:
    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_dataframe(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = None
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            bars = await fetch_bars_yfinance("AAPL", date(2024, 1, 1), date(2024, 1, 31))
            assert bars == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_fetch_error(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("network error")
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            bars = await fetch_bars_yfinance("AAPL", date(2024, 1, 1), date(2024, 1, 31))
            assert bars == []


class TestFetchBars:
    @pytest.mark.asyncio
    async def test_falls_back_to_yfinance_on_alpaca_failure(self):
        yf_bars = [{"symbol": "AAPL", "date": date(2024, 1, 15), "open": Decimal("100"),
                     "high": Decimal("110"), "low": Decimal("90"), "close": Decimal("105"),
                     "volume": 1000, "vwap": None}]

        with patch("src.data.market_data.fetch_bars_alpaca", side_effect=Exception("api error")):
            with patch("src.data.market_data.fetch_bars_yfinance", return_value=yf_bars) as mock_yf:
                bars = await fetch_bars("AAPL", date(2024, 1, 1), date(2024, 1, 31))
                assert len(bars) == 1
                mock_yf.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_when_alpaca_returns_empty(self):
        with patch("src.data.market_data.fetch_bars_alpaca", return_value=[]):
            with patch("src.data.market_data.fetch_bars_yfinance", return_value=[]) as mock_yf:
                bars = await fetch_bars("AAPL", date(2024, 1, 1), date(2024, 1, 31))
                assert bars == []
                mock_yf.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_alpaca_when_successful(self):
        alpaca_bars = [{"symbol": "AAPL", "date": date(2024, 1, 15)}]

        with patch("src.data.market_data.fetch_bars_alpaca", return_value=alpaca_bars):
            with patch("src.data.market_data.fetch_bars_yfinance") as mock_yf:
                bars = await fetch_bars("AAPL", date(2024, 1, 1), date(2024, 1, 31))
                assert len(bars) == 1
                mock_yf.assert_not_called()


class TestStoreBars:
    @pytest.mark.asyncio
    async def test_stores_valid_bars(self):
        bars = [
            {"symbol": "AAPL", "date": date(2024, 1, 15),
             "open": Decimal("100"), "high": Decimal("110"),
             "low": Decimal("90"), "close": Decimal("105"),
             "volume": 1000, "vwap": Decimal("102")},
        ]

        mock_conn = AsyncMock()
        pool = _mock_pool(mock_conn)

        count = await store_bars(pool, bars)
        assert count == 1
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_invalid_bars(self):
        bars = [
            {"symbol": "BAD", "date": date(2024, 1, 15),
             "open": Decimal("-999"), "high": Decimal("110"),
             "low": Decimal("90"), "close": Decimal("105"),
             "volume": 1000, "vwap": None},
        ]

        mock_pool = AsyncMock()
        count = await store_bars(mock_pool, bars)
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_bars_returns_zero(self):
        mock_pool = AsyncMock()
        count = await store_bars(mock_pool, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_continues_on_insert_error(self):
        bars = [
            {"symbol": "AAPL", "date": date(2024, 1, 15),
             "open": Decimal("100"), "high": Decimal("110"),
             "low": Decimal("90"), "close": Decimal("105"),
             "volume": 1000, "vwap": None},
            {"symbol": "MSFT", "date": date(2024, 1, 15),
             "open": Decimal("200"), "high": Decimal("210"),
             "low": Decimal("190"), "close": Decimal("205"),
             "volume": 2000, "vwap": None},
        ]

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = [Exception("db error"), None]
        pool = _mock_pool(mock_conn)

        count = await store_bars(pool, bars)
        # / first fails, second succeeds
        assert count == 1


class TestBackfill:
    @pytest.mark.asyncio
    async def test_incremental_backfill_skips_up_to_date(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"max_date": date.today()}
        pool = _mock_pool(mock_conn)

        with patch("src.data.market_data.fetch_bars") as mock_fetch:
            results = await backfill(pool, ["AAPL"], years=5)
            assert results["AAPL"] == 0
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_backfill_fetches_from_scratch(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"max_date": None}
        pool = _mock_pool(mock_conn)

        bars = [{"symbol": "AAPL", "date": date(2024, 1, 15),
                 "open": Decimal("100"), "high": Decimal("110"),
                 "low": Decimal("90"), "close": Decimal("105"),
                 "volume": 1000, "vwap": None}]

        with patch("src.data.market_data.fetch_bars", return_value=bars):
            with patch("src.data.market_data.store_bars", return_value=1) as mock_store:
                results = await backfill(pool, ["AAPL"], years=5)
                assert results["AAPL"] == 1
                mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_backfill_continues_on_symbol_failure(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"max_date": None}
        pool = _mock_pool(mock_conn)

        call_count = 0

        async def mock_fetch(sym, start, end):
            nonlocal call_count
            call_count += 1
            if sym == "AAPL":
                raise Exception("api error")
            return [{"symbol": sym, "date": date(2024, 1, 15),
                     "open": Decimal("100"), "high": Decimal("110"),
                     "low": Decimal("90"), "close": Decimal("105"),
                     "volume": 1000, "vwap": None}]

        with patch("src.data.market_data.fetch_bars", side_effect=mock_fetch):
            with patch("src.data.market_data.store_bars", return_value=1):
                results = await backfill(pool, ["AAPL", "MSFT"], years=5)
                assert results["AAPL"] == 0
                assert results["MSFT"] == 1


class TestFetchLatestQuote:
    @pytest.mark.asyncio
    async def test_fetches_equity_quote(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "trade": {"p": 150.25, "t": "2024-01-15T15:30:00Z"}
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_data.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_latest_quote("AAPL")
            assert result["symbol"] == "AAPL"
            assert result["price"] == Decimal("150.25")

    @pytest.mark.asyncio
    async def test_fetches_crypto_quote(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "trades": {"BTC/USD": {"p": 42000.50, "t": "2024-01-15T15:30:00Z"}}
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.data.market_data.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_latest_quote("BTC-USD")
            assert result["symbol"] == "BTC-USD"
            assert result["price"] == Decimal("42000.50")
