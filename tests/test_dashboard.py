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
                         "fundamentals", "dcf", "price_history", "insider_trades", "evolution"}
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
        # / health v2 makes many queries — use return_value for all fetchrow/fetch calls
        conn.fetchrow.return_value = None
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "db_connected" in data
        assert "storage" in data
        assert "connections" in data
        assert "cycles" in data
        assert "sources" in data
        assert "recent_errors" in data
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_synthesis_endpoint_returns_latest(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetchrow.return_value = None
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/synthesis")
        assert resp.status_code == 200
        assert resp.json() is None
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_synthesis_history_returns_list(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/synthesis/history?days=3")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        dashboard._pool = None

    @pytest.mark.asyncio
    async def test_analysis_includes_evolution(self):
        from src.dashboard import app as dashboard
        pool, conn = _mock_pool()
        dashboard._pool = pool
        conn.fetchrow.return_value = None
        conn.fetch.return_value = []
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test") as c:
            resp = await c.get("/api/analysis/AAPL")
        data = resp.json()
        assert "evolution" in data
        assert data["evolution"] == []
        dashboard._pool = None


# / helper: set up httpx async client against dashboard app
async def _client():
    from httpx import AsyncClient, ASGITransport
    from src.dashboard import app as dashboard
    return AsyncClient(transport=ASGITransport(app=dashboard.app), base_url="http://test")


# / helper: mock _query and _query_one at module level
# / avoids dealing with asyncpg Record mock — endpoints get plain dicts
def _patch_query(query_results=None, query_one_result=None):
    if query_results is None:
        query_results = []

    async def mock_query(sql, *args):
        if callable(query_results):
            return await query_results(sql, *args)
        return query_results

    async def mock_query_one(sql, *args):
        if callable(query_one_result):
            return await query_one_result(sql, *args)
        return query_one_result

    return (
        patch("src.dashboard.app._query", new=mock_query),
        patch("src.dashboard.app._query_one", new=mock_query_one),
    )


# / helper: mock broker via _get_broker
def _mock_broker(balance=None, positions=None, error=None):
    if error:
        return patch("src.dashboard.app._get_broker", side_effect=error)
    broker = AsyncMock()
    if balance:
        broker.get_account_balance.return_value = balance
    if positions is not None:
        broker.get_positions.return_value = positions
    return patch("src.dashboard.app._get_broker", return_value=broker)


def _make_balance(equity=100000.0, cash=50000.0, buying_power=200000.0):
    b = MagicMock()
    b.equity = equity
    b.cash = cash
    b.buying_power = buying_power
    return b


def _make_position(symbol="AAPL", side="long", qty=10.0, entry=175.0,
                   mv=1820.0, pnl=70.0, price=182.0):
    p = MagicMock()
    p.symbol = symbol
    p.side = side
    p.qty = qty
    p.avg_entry_price = entry
    p.market_value = mv
    p.unrealized_pnl = pnl
    p.current_price = price
    return p


class TestPortfolioEndpoint:
    @pytest.mark.asyncio
    async def test_portfolio_returns_broker_data(self):
        from src.dashboard import app as dashboard
        pos = _make_position()
        pq, pqo = _patch_query(query_results=[])
        with _mock_broker(balance=_make_balance(), positions=[pos]), pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/portfolio")
        data = resp.json()
        assert resp.status_code == 200
        assert data["equity"] == 100000.0
        assert data["cash"] == 50000.0
        assert data["buying_power"] == 200000.0
        assert data["positions_count"] == 1
        assert data["daily_pnl"] == 70.0
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "AAPL"
        assert data["positions"][0]["current_price"] == 182.0
        dashboard._broker = None

    @pytest.mark.asyncio
    async def test_portfolio_fallback_on_broker_error(self):
        from src.dashboard import app as dashboard
        fallback_row = {"symbol": "MSFT", "side": "buy", "qty": 5, "price": 400.0, "strategy_id": "s1", "created_at": "2026-03-26"}
        pq, pqo = _patch_query(query_results=[fallback_row])
        with _mock_broker(error=Exception("no keys")), pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/portfolio")
        data = resp.json()
        assert resp.status_code == 200
        assert data["positions_count"] == 0
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "MSFT"
        assert data["trades_today"] == []
        dashboard._broker = None

    @pytest.mark.asyncio
    async def test_portfolio_empty_positions(self):
        from src.dashboard import app as dashboard
        pq, pqo = _patch_query(query_results=[])
        with _mock_broker(balance=_make_balance(50000, 50000, 100000), positions=[]), pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/portfolio")
        data = resp.json()
        assert data["positions_count"] == 0
        assert data["daily_pnl"] == 0
        assert data["positions"] == []
        dashboard._broker = None

    @pytest.mark.asyncio
    async def test_portfolio_multiple_positions_pnl_sum(self):
        from src.dashboard import app as dashboard
        p1 = _make_position("AAPL", pnl=50.0)
        p2 = _make_position("MSFT", pnl=-20.0)
        pq, pqo = _patch_query(query_results=[])
        with _mock_broker(balance=_make_balance(), positions=[p1, p2]), pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/portfolio")
        data = resp.json()
        assert data["daily_pnl"] == 30.0
        assert data["positions_count"] == 2
        dashboard._broker = None


class TestEquityHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_equity_history_returns_data(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "timestamp": [1711400000, 1711400300],
            "equity": [100000, 100500],
            "profit_loss": [0, 500],
            "base_value": 100000,
        }
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp

        with patch("src.data.alpaca_client.alpaca_base_url", return_value="https://paper-api.alpaca.markets"), \
             patch("src.data.alpaca_client.alpaca_headers", return_value={"APCA-API-KEY-ID": "x"}), \
             patch("src.data.alpaca_client.get_alpaca_client", return_value=mock_http):
            async with await _client() as c:
                resp = await c.get("/api/equity-history?period=1D&timeframe=5Min")
        data = resp.json()
        assert resp.status_code == 200
        assert data["timestamps"] == [1711400000, 1711400300]
        assert data["equity"] == [100000, 100500]
        assert data["profit_loss"] == [0, 500]
        assert data["base_value"] == 100000

    @pytest.mark.asyncio
    async def test_equity_history_fallback_on_error(self):
        with patch("src.data.alpaca_client.alpaca_base_url", return_value="https://paper-api.alpaca.markets"), \
             patch("src.data.alpaca_client.alpaca_headers", return_value={}), \
             patch("src.data.alpaca_client.get_alpaca_client", side_effect=Exception("network error")):
            async with await _client() as c:
                resp = await c.get("/api/equity-history")
        data = resp.json()
        assert resp.status_code == 200
        assert data["timestamps"] == []
        assert data["equity"] == []
        assert data["profit_loss"] == []
        assert data["base_value"] == 100000

    @pytest.mark.asyncio
    async def test_equity_history_missing_fields_default(self):
        # / alpaca response missing base_value falls back to 100000
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"timestamp": [], "equity": [], "profit_loss": []}
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp

        with patch("src.data.alpaca_client.alpaca_base_url", return_value="https://paper-api.alpaca.markets"), \
             patch("src.data.alpaca_client.alpaca_headers", return_value={}), \
             patch("src.data.alpaca_client.get_alpaca_client", return_value=mock_http):
            async with await _client() as c:
                resp = await c.get("/api/equity-history")
        data = resp.json()
        assert data["base_value"] == 100000


class TestPositionsEndpoint:
    @pytest.mark.asyncio
    async def test_positions_returns_broker_data(self):
        from src.dashboard import app as dashboard
        pos = _make_position("TSLA", "long", 5.0, 250.0, 1350.0, 100.0, 270.0)
        with _mock_broker(positions=[pos]):
            async with await _client() as c:
                resp = await c.get("/api/positions")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["symbol"] == "TSLA"
        assert data[0]["entry_price"] == 250.0
        assert data[0]["unrealized_pl"] == 100.0
        dashboard._broker = None

    @pytest.mark.asyncio
    async def test_positions_empty(self):
        from src.dashboard import app as dashboard
        with _mock_broker(positions=[]):
            async with await _client() as c:
                resp = await c.get("/api/positions")
        assert resp.status_code == 200
        assert resp.json() == []
        dashboard._broker = None

    @pytest.mark.asyncio
    async def test_positions_fallback_on_error(self):
        from src.dashboard import app as dashboard
        fallback_row = {"symbol": "NVDA", "side": "buy", "qty": 3, "entry_price": 800.0, "strategy_id": "s2", "created_at": "2026-03-25"}
        pq, pqo = _patch_query(query_results=[fallback_row])
        with _mock_broker(error=Exception("no creds")), pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/positions")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["symbol"] == "NVDA"
        dashboard._broker = None


class TestTradesEndpoint:
    @pytest.mark.asyncio
    async def test_trades_returns_data(self):
        rows = [
            {"id": 1, "symbol": "AAPL", "side": "buy", "qty": 10, "price": 180.0, "created_at": "2026-03-26"},
            {"id": 2, "symbol": "MSFT", "side": "sell", "qty": 5, "price": 410.0, "created_at": "2026-03-25"},
        ]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/trades?limit=10&offset=0")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 2
        assert data[0]["symbol"] == "AAPL"
        assert data[1]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_trades_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/trades")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_trades_with_symbol_filter(self):
        # / verify symbol query param routes to the symbol-filtered sql branch
        calls = []

        async def track_query(sql, *args):
            calls.append((sql, args))
            return [{"id": 1, "symbol": "GOOG", "side": "buy", "qty": 2, "price": 170.0, "created_at": "2026-03-26"}]

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/trades?symbol=GOOG")
        data = resp.json()
        assert resp.status_code == 200
        assert data[0]["symbol"] == "GOOG"
        assert any("GOOG" in args for _, args in calls)

    @pytest.mark.asyncio
    async def test_trades_limit_clamped(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/trades?limit=9999")
        assert resp.status_code == 200
        # / verify clamped limit=500 was passed
        assert any(500 in args for args in calls)

    @pytest.mark.asyncio
    async def test_trades_pool_none(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        async with await _client() as c:
            resp = await c.get("/api/trades")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_trades_offset_clamped_to_zero(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/trades?offset=-5")
        assert resp.status_code == 200
        # / negative offset clamped to 0
        assert any(0 in args for args in calls)


class TestStrategiesEndpoint:
    @pytest.mark.asyncio
    async def test_strategies_returns_scores(self):
        rows = [{"strategy_id": "s1", "sharpe_ratio": 1.5, "win_rate": 0.6, "total_pnl": 5000}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategies")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s1"
        assert data[0]["sharpe_ratio"] == 1.5

    @pytest.mark.asyncio
    async def test_strategies_fallback_to_trade_log(self):
        # / first call returns empty (strategy_scores), second returns fallback
        call_count = {"n": 0}

        async def sequential_query(sql, *args):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return []
            return [{"strategy_id": "s2", "total_trades": 10, "wins": 6, "losses": 4,
                     "avg_pnl": 50, "total_pnl": 500, "win_rate": 0.6, "last_trade_at": "2026-03-26"}]

        pq, pqo = _patch_query(query_results=sequential_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategies")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s2"
        assert data[0]["total_trades"] == 10

    @pytest.mark.asyncio
    async def test_strategies_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategies")
        assert resp.status_code == 200
        assert resp.json() == []


class TestEvolutionEndpoint:
    @pytest.mark.asyncio
    async def test_evolution_returns_data(self):
        rows = [{"generation": 5, "action": "mutate", "strategy_id": "s1",
                 "reason": "low sharpe", "details": {}, "created_at": "2026-03-26T00:00:00"}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/evolution")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["generation"] == 5
        assert data[0]["action"] == "mutate"

    @pytest.mark.asyncio
    async def test_evolution_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/evolution")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_evolution_pool_none(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        async with await _client() as c:
            resp = await c.get("/api/evolution")
        assert resp.status_code == 200
        assert resp.json() == []


class TestInsiderEndpoint:
    @pytest.mark.asyncio
    async def test_insider_returns_data(self):
        rows = [{"filing_date": "2026-03-20", "insider_name": "Tim Cook", "insider_title": "CEO",
                 "transaction_type": "S-Sale", "shares": 50000, "price_per_share": 180.0, "total_value": 9000000}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/insider/AAPL")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["insider_name"] == "Tim Cook"
        assert data[0]["shares"] == 50000

    @pytest.mark.asyncio
    async def test_insider_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/insider/ZZZZ")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_insider_uppercases_symbol(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/insider/aapl")
        assert resp.status_code == 200
        assert any("AAPL" in args for args in calls)


class TestSignalsEndpoint:
    @pytest.mark.asyncio
    async def test_signals_returns_data(self):
        rows = [{"id": 1, "symbol": "AAPL", "strategy_id": "s1", "signal": "buy",
                 "confidence": 0.85, "created_at": "2026-03-26"}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/signals")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"
        assert data[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_signals_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/signals")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_signals_limit_clamped(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/signals?limit=999")
        assert any(500 in args for args in calls)

    @pytest.mark.asyncio
    async def test_signals_custom_limit(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/signals?limit=10")
        assert any(10 in args for args in calls)

    @pytest.mark.asyncio
    async def test_signals_limit_minimum_one(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/signals?limit=0")
        assert any(1 in args for args in calls)


class TestIndicatorsEndpoint:
    @pytest.mark.asyncio
    async def test_indicators_returns_data(self):
        rows = [{"date": "2026-03-26", "rsi14": 45.2, "macd": 1.5, "macd_signal": 1.2,
                 "macd_histogram": 0.3, "adx": 25.0, "sma20": 178.0, "sma50": 175.0,
                 "bb_upper": 185.0, "bb_middle": 178.0, "bb_lower": 171.0, "atr": 3.5, "timeframe": "1Day"}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/indicators/AAPL")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["rsi14"] == 45.2
        assert data[0]["macd"] == 1.5
        assert data[0]["timeframe"] == "1Day"

    @pytest.mark.asyncio
    async def test_indicators_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/indicators/ZZZZ")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_indicators_custom_timeframe(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/indicators/AAPL?timeframe=1Hour&limit=30")
        assert any("1Hour" in args and 30 in args for args in calls)

    @pytest.mark.asyncio
    async def test_indicators_limit_clamped(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/indicators/AAPL?limit=9999")
        assert any(250 in args for args in calls)


class TestIntradayEndpoint:
    @pytest.mark.asyncio
    async def test_intraday_returns_data(self):
        rows = [{"timestamp": "2026-03-26T10:00:00", "open": 180.0, "high": 182.0,
                 "low": 179.0, "close": 181.5, "volume": 50000, "vwap": 180.8}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/intraday/AAPL")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["close"] == 181.5
        assert data[0]["vwap"] == 180.8

    @pytest.mark.asyncio
    async def test_intraday_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/intraday/ZZZZ")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_intraday_custom_params(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/intraday/TSLA?days=5&timeframe=1Hour")
        assert any("TSLA" in args and "1Hour" in args and "5" in args for args in calls)

    @pytest.mark.asyncio
    async def test_intraday_days_clamped(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/intraday/AAPL?days=999")
        # / days clamped to 60, passed as string "60"
        assert any("60" in args for args in calls)


class TestQuantMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_quant_metrics_returns_data(self):
        rows = [{"strategy_id": "s1", "sharpe_ratio": 1.8, "win_rate": 0.65, "max_drawdown": -0.08}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/quant-metrics/AAPL")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["sharpe_ratio"] == 1.8

    @pytest.mark.asyncio
    async def test_quant_metrics_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/quant-metrics/ZZZZ")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_quant_metrics_pool_none(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        async with await _client() as c:
            resp = await c.get("/api/quant-metrics/AAPL")
        assert resp.status_code == 200
        assert resp.json() == []


class TestStrategyPositionsEndpoint:
    @pytest.mark.asyncio
    async def test_strategy_positions_all(self):
        rows = [
            {"strategy_id": "s1", "symbol": "AAPL", "qty": 10, "avg_entry_price": 175.0, "updated_at": "2026-03-26"},
            {"strategy_id": "s2", "symbol": "MSFT", "qty": 5, "avg_entry_price": 400.0, "updated_at": "2026-03-25"},
        ]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-positions")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 2
        assert data[0]["strategy_id"] == "s1"
        assert data[1]["symbol"] == "MSFT"

    @pytest.mark.asyncio
    async def test_strategy_positions_by_symbol(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return [{"strategy_id": "s1", "symbol": "AAPL", "qty": 10, "avg_entry_price": 175.0, "updated_at": "2026-03-26"}]

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-positions?symbol=AAPL")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert any("AAPL" in args for args in calls)

    @pytest.mark.asyncio
    async def test_strategy_positions_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-positions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_strategy_positions_pool_none(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        async with await _client() as c:
            resp = await c.get("/api/strategy-positions")
        assert resp.status_code == 200
        assert resp.json() == []


class TestStrategyEvaluationsEndpoint:
    @pytest.mark.asyncio
    async def test_evaluations_returns_data(self):
        rows = [{"id": 1, "strategy_id": "s1", "sharpe": 1.5, "win_rate": 0.6, "created_at": "2026-03-26"}]
        pq, pqo = _patch_query(query_results=rows)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-evaluations")
        data = resp.json()
        assert resp.status_code == 200
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s1"

    @pytest.mark.asyncio
    async def test_evaluations_empty(self):
        pq, pqo = _patch_query()
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-evaluations")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_evaluations_limit_clamped(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-evaluations?limit=999")
        # / clamped to 100
        assert any(100 in args for args in calls)

    @pytest.mark.asyncio
    async def test_evaluations_custom_limit(self):
        calls = []

        async def track_query(sql, *args):
            calls.append(args)
            return []

        pq, pqo = _patch_query(query_results=track_query)
        with pq, pqo:
            async with await _client() as c:
                resp = await c.get("/api/strategy-evaluations?limit=5")
        assert any(5 in args for args in calls)

    @pytest.mark.asyncio
    async def test_evaluations_pool_none(self):
        from src.dashboard import app as dashboard
        dashboard._pool = None
        async with await _client() as c:
            resp = await c.get("/api/strategy-evaluations")
        assert resp.status_code == 200
        assert resp.json() == []
