# / tests for fundamentals fetching, sector averages, validation, storage

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.fundamentals import (
    _compute_fcf_margin,
    _compute_sector_averages,
    _safe_decimal,
    fetch_all_fundamentals,
    fetch_fundamentals,
    store_fundamentals,
)


def _mock_pool(mock_conn):
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool


class TestSafeDecimal:
    def test_converts_int(self):
        assert _safe_decimal(42) == Decimal("42")

    def test_converts_float(self):
        result = _safe_decimal(3.14)
        assert isinstance(result, Decimal)

    def test_converts_string(self):
        assert _safe_decimal("100.5") == Decimal("100.5")

    def test_returns_none_for_none(self):
        assert _safe_decimal(None) is None

    def test_returns_none_for_nan(self):
        assert _safe_decimal(float("nan")) is None

    def test_returns_none_for_inf(self):
        assert _safe_decimal(float("inf")) is None

    def test_returns_none_for_invalid_string(self):
        assert _safe_decimal("not_a_number") is None


class TestComputeFcfMargin:
    def test_computes_margin(self):
        info = {"freeCashflow": 1000000, "totalRevenue": 5000000}
        result = _compute_fcf_margin(info)
        assert result == Decimal("0.2")

    def test_returns_none_when_no_fcf(self):
        info = {"totalRevenue": 5000000}
        assert _compute_fcf_margin(info) is None

    def test_returns_none_when_no_revenue(self):
        info = {"freeCashflow": 1000000}
        assert _compute_fcf_margin(info) is None

    def test_returns_none_when_revenue_zero(self):
        info = {"freeCashflow": 1000000, "totalRevenue": 0}
        assert _compute_fcf_margin(info) is None

    def test_negative_fcf(self):
        info = {"freeCashflow": -500000, "totalRevenue": 5000000}
        result = _compute_fcf_margin(info)
        assert result == Decimal("-0.1")


class TestComputeSectorAverages:
    def test_computes_averages_per_sector(self):
        data = [
            {"symbol": "AAPL", "sector": "Technology", "pe_ratio": Decimal("30"), "ps_ratio": Decimal("8")},
            {"symbol": "MSFT", "sector": "Technology", "pe_ratio": Decimal("35"), "ps_ratio": Decimal("12")},
            {"symbol": "JPM", "sector": "Finance", "pe_ratio": Decimal("12"), "ps_ratio": Decimal("3")},
        ]
        _compute_sector_averages(data)

        # / tech avg pe = (30+35)/2 = 32.5
        assert data[0]["sector_pe_avg"] == Decimal("32.5")
        assert data[1]["sector_pe_avg"] == Decimal("32.5")
        # / finance avg pe = 12
        assert data[2]["sector_pe_avg"] == Decimal("12")

    def test_handles_none_values(self):
        data = [
            {"symbol": "AAPL", "sector": "Technology", "pe_ratio": Decimal("30"), "ps_ratio": None},
            {"symbol": "MSFT", "sector": "Technology", "pe_ratio": None, "ps_ratio": Decimal("12")},
        ]
        _compute_sector_averages(data)
        # / only one pe value, so avg = 30
        assert data[0]["sector_pe_avg"] == Decimal("30")
        # / only one ps value, so avg = 12
        assert data[0]["sector_ps_avg"] == Decimal("12")

    def test_empty_data(self):
        data: list = []
        _compute_sector_averages(data)
        # / should not crash


class TestFetchFundamentals:
    @pytest.mark.asyncio
    async def test_returns_data_on_success(self):
        mock_info = {
            "regularMarketPrice": 150.0,
            "trailingPE": 25.0,
            "forwardPE": 22.0,
            "priceToSalesTrailing12Months": 8.0,
            "pegRatio": 1.5,
            "revenueGrowth": 0.15,
            "freeCashflow": 1000000,
            "totalRevenue": 5000000,
            "debtToEquity": 50.0,
            "sector": "Technology",
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = await fetch_fundamentals("AAPL")
            assert result is not None
            assert result["symbol"] == "AAPL"
            assert result["pe_ratio"] == Decimal("25.0")
            assert result["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_returns_none_on_no_data(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = await fetch_fundamentals("FAKE")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = await fetch_fundamentals("AAPL")
            assert result is None


class TestFetchAllFundamentals:
    @pytest.mark.asyncio
    async def test_fetches_all_symbols(self):
        results = [
            {"symbol": "AAPL", "date": date.today(), "pe_ratio": Decimal("25"),
             "ps_ratio": Decimal("8"), "sector": "Technology",
             "pe_forward": None, "peg_ratio": None, "revenue_growth_1y": None,
             "revenue_growth_3y": None, "fcf_margin": None, "debt_to_equity": None,
             "sector_pe_avg": None, "sector_ps_avg": None},
            {"symbol": "MSFT", "date": date.today(), "pe_ratio": Decimal("30"),
             "ps_ratio": Decimal("10"), "sector": "Technology",
             "pe_forward": None, "peg_ratio": None, "revenue_growth_1y": None,
             "revenue_growth_3y": None, "fcf_margin": None, "debt_to_equity": None,
             "sector_pe_avg": None, "sector_ps_avg": None},
        ]

        call_idx = 0

        async def mock_fetch(sym):
            nonlocal call_idx
            r = results[call_idx]
            call_idx += 1
            return r

        with patch("src.data.fundamentals.fetch_fundamentals", side_effect=mock_fetch):
            with patch("asyncio.sleep", return_value=None):
                data = await fetch_all_fundamentals(["AAPL", "MSFT"])
                assert len(data) == 2
                # / sector averages should be computed
                assert data[0]["sector_pe_avg"] is not None

    @pytest.mark.asyncio
    async def test_skips_failed_symbols(self):
        async def mock_fetch(sym):
            if sym == "AAPL":
                return {"symbol": "AAPL", "date": date.today(), "pe_ratio": Decimal("25"),
                        "ps_ratio": None, "sector": "Tech",
                        "pe_forward": None, "peg_ratio": None, "revenue_growth_1y": None,
                        "revenue_growth_3y": None, "fcf_margin": None, "debt_to_equity": None,
                        "sector_pe_avg": None, "sector_ps_avg": None}
            return None

        with patch("src.data.fundamentals.fetch_fundamentals", side_effect=mock_fetch):
            with patch("asyncio.sleep", return_value=None):
                data = await fetch_all_fundamentals(["AAPL", "FAKE"])
                assert len(data) == 1


class TestStoreFundamentals:
    @pytest.mark.asyncio
    async def test_stores_valid_data(self):
        data = [
            {"symbol": "AAPL", "date": date.today(),
             "pe_ratio": Decimal("25"), "pe_forward": Decimal("22"),
             "ps_ratio": Decimal("8"), "peg_ratio": Decimal("1.5"),
             "revenue_growth_1y": Decimal("0.15"), "revenue_growth_3y": None,
             "fcf_margin": Decimal("0.2"), "debt_to_equity": Decimal("50"),
             "sector": "Technology", "sector_pe_avg": Decimal("30"),
             "sector_ps_avg": Decimal("10")},
        ]

        mock_conn = AsyncMock()
        pool = _mock_pool(mock_conn)

        count = await store_fundamentals(pool, data)
        assert count == 1
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_invalid_data(self):
        # / pe_ratio out of bounds
        data = [
            {"symbol": "BAD", "date": date.today(),
             "pe_ratio": Decimal("99999"), "pe_forward": None,
             "ps_ratio": None, "peg_ratio": None,
             "revenue_growth_1y": None, "revenue_growth_3y": None,
             "fcf_margin": None, "debt_to_equity": None,
             "sector": "Unknown", "sector_pe_avg": None,
             "sector_ps_avg": None},
        ]

        mock_pool = AsyncMock()
        count = await store_fundamentals(mock_pool, data)
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_data_returns_zero(self):
        mock_pool = AsyncMock()
        count = await store_fundamentals(mock_pool, [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_continues_on_insert_error(self):
        data = [
            {"symbol": "AAPL", "date": date.today(),
             "pe_ratio": Decimal("25"), "pe_forward": None,
             "ps_ratio": None, "peg_ratio": None,
             "revenue_growth_1y": None, "revenue_growth_3y": None,
             "fcf_margin": None, "debt_to_equity": None,
             "sector": "Tech", "sector_pe_avg": None, "sector_ps_avg": None},
            {"symbol": "MSFT", "date": date.today(),
             "pe_ratio": Decimal("30"), "pe_forward": None,
             "ps_ratio": None, "peg_ratio": None,
             "revenue_growth_1y": None, "revenue_growth_3y": None,
             "fcf_margin": None, "debt_to_equity": None,
             "sector": "Tech", "sector_pe_avg": None, "sector_ps_avg": None},
        ]

        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = [Exception("db error"), None]
        pool = _mock_pool(mock_conn)

        count = await store_fundamentals(pool, data)
        assert count == 1
