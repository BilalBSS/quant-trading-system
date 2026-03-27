# / tests for dashboard api endpoints

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date, datetime
from decimal import Decimal

from src.dashboard.app import _serialize, _serialize_one


class TestSerialize:
    def test_serializes_decimal(self):
        row = {"price": Decimal("182.40")}
        result = _serialize_one(row)
        assert result["price"] == "182.40"

    def test_serializes_date(self):
        row = {"date": date(2026, 3, 26)}
        result = _serialize_one(row)
        assert result["date"] == "2026-03-26"

    def test_serializes_datetime(self):
        row = {"created_at": datetime(2026, 3, 26, 14, 30)}
        result = _serialize_one(row)
        assert "2026-03-26" in result["created_at"]

    def test_preserves_primitives(self):
        row = {"count": 5, "name": "test", "active": True, "note": None}
        result = _serialize_one(row)
        assert result["count"] == 5
        assert result["name"] == "test"
        assert result["active"] is True
        assert result["note"] is None

    def test_none_returns_none(self):
        assert _serialize_one(None) is None

    def test_serialize_list(self):
        rows = [{"a": 1}, {"a": 2}]
        result = _serialize(rows)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_serialize_empty_list(self):
        assert _serialize([]) == []

    def test_preserves_dict_jsonb(self):
        row = {"details": {"ai_consensus": "bullish", "pe_ratio": 18.5}}
        result = _serialize_one(row)
        assert isinstance(result["details"], dict)
        assert result["details"]["ai_consensus"] == "bullish"
        assert result["details"]["pe_ratio"] == 18.5

    def test_preserves_list_jsonb(self):
        row = {"items": [1, "two", 3.0]}
        result = _serialize_one(row)
        assert isinstance(result["items"], list)
        assert result["items"] == [1, "two", 3.0]

    def test_preserves_nested_jsonb(self):
        row = {"assumptions": {"growth_rate": 0.05, "ranges": {"min": 0.01, "max": 0.10}}}
        result = _serialize_one(row)
        assert result["assumptions"]["growth_rate"] == 0.05
        assert result["assumptions"]["ranges"]["max"] == 0.10

    def test_unknown_type_becomes_string(self):
        # / types that are not dict, list, primitive, date, or decimal
        row = {"data": frozenset([1, 2])}
        result = _serialize_one(row)
        assert isinstance(result["data"], str)


# / mock pool helper (same pattern as agent tests)
def _mock_pool(rows=None, row=None):
    mock_conn = AsyncMock()
    if rows is not None:
        mock_conn.fetch.return_value = [MagicMock(**{"items.return_value": list(r.items()), "keys.return_value": list(r.keys()), "__iter__": lambda s: iter(r.items())}) for r in rows]
    else:
        mock_conn.fetch.return_value = []
    if row is not None:
        mock_conn.fetchrow.return_value = MagicMock(**{"items.return_value": list(row.items()), "keys.return_value": list(row.keys()), "__iter__": lambda s: iter(row.items())})
    else:
        mock_conn.fetchrow.return_value = None
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_conn
    mock_ctx.__aexit__.return_value = False
    pool = MagicMock()
    pool.acquire.return_value = mock_ctx
    return pool, mock_conn


class TestAnalysisEndpoint:
    @pytest.mark.asyncio
    async def test_returns_all_keys(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        # / mock asyncpg Record objects
        conn.fetchrow.return_value = None
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/analysis/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {"score", "signals", "trades", "sentiment", "social_sentiment",
                         "fundamentals", "dcf", "price_history", "insider_trades"}
        assert set(data.keys()) == expected_keys
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_empty_symbol_returns_none_and_empty_lists(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetchrow.return_value = None
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/analysis/NONEXISTENT")
        data = resp.json()
        assert data["score"] is None
        assert data["fundamentals"] is None
        assert data["dcf"] is None
        assert data["signals"] == []
        assert data["trades"] == []
        assert data["sentiment"] == []
        assert data["insider_trades"] == []
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/analysis/AAPL")
        data = resp.json()
        assert data["score"] is None
        assert data["signals"] == []
        assert data["insider_trades"] == []

    @pytest.mark.asyncio
    async def test_symbols_endpoint(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/symbols")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetchrow.side_effect = [
            MagicMock(**{"items.return_value": [("ok", 1)], "__iter__": lambda s: iter([("ok", 1)])}),
            None, None, None,
            MagicMock(**{"items.return_value": [("size_bytes", 1024000)], "__iter__": lambda s: iter([("size_bytes", 1024000)])}),
        ]
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "db_connected" in data
        dashboard._pool = None
