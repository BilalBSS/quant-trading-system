# / tests for paper broker

from __future__ import annotations

import pytest

from src.brokers.base import AccountBalance, Order, Position
from src.brokers.paper_broker import PaperBroker


@pytest.fixture
def broker():
    b = PaperBroker(initial_cash=100_000.0)
    b.set_prices({"AAPL": 150.0, "MSFT": 300.0, "GOOG": 140.0})
    return b


class TestGetPrice:
    @pytest.mark.asyncio
    async def test_returns_set_price(self, broker):
        price = await broker.get_price("AAPL")
        assert price == 150.0

    @pytest.mark.asyncio
    async def test_raises_on_missing(self, broker):
        with pytest.raises(ValueError, match="no price"):
            await broker.get_price("FAKE")


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_market_buy(self, broker):
        order = await broker.place_order("AAPL", 10, "buy")
        assert order.status == "filled"
        assert order.filled_qty == 10
        assert order.filled_price == 150.0
        assert broker.cash == pytest.approx(100_000 - 1500)

    @pytest.mark.asyncio
    async def test_market_sell(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        order = await broker.place_order("AAPL", 5, "sell")
        assert order.status == "filled"
        assert order.filled_qty == 5
        assert broker.cash == pytest.approx(100_000 - 1500 + 750)

    @pytest.mark.asyncio
    async def test_insufficient_cash(self, broker):
        # / try to buy $200k worth
        order = await broker.place_order("MSFT", 700, "buy")
        assert order.status == "rejected"
        assert order.details["reason"] == "insufficient_cash"

    @pytest.mark.asyncio
    async def test_insufficient_position(self, broker):
        order = await broker.place_order("AAPL", 10, "sell")
        assert order.status == "rejected"
        assert order.details["reason"] == "insufficient_position"

    @pytest.mark.asyncio
    async def test_invalid_qty(self, broker):
        with pytest.raises(ValueError, match="positive"):
            await broker.place_order("AAPL", 0, "buy")

    @pytest.mark.asyncio
    async def test_invalid_side(self, broker):
        with pytest.raises(ValueError, match="buy or sell"):
            await broker.place_order("AAPL", 10, "short")

    @pytest.mark.asyncio
    async def test_no_price_rejected(self, broker):
        order = await broker.place_order("FAKE", 10, "buy")
        assert order.status == "rejected"

    @pytest.mark.asyncio
    async def test_limit_buy_not_fillable(self, broker):
        # / price is 150, limit at 140 — shouldn't fill
        order = await broker.place_order("AAPL", 10, "buy", order_type="limit", limit_price=140.0)
        assert order.status == "pending"

    @pytest.mark.asyncio
    async def test_limit_buy_fillable(self, broker):
        # / price is 150, limit at 160 — should fill at 160
        order = await broker.place_order("AAPL", 10, "buy", order_type="limit", limit_price=160.0)
        assert order.status == "filled"
        assert order.filled_price == 160.0

    @pytest.mark.asyncio
    async def test_sell_removes_position(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        await broker.place_order("AAPL", 10, "sell")
        assert "AAPL" not in broker.positions

    @pytest.mark.asyncio
    async def test_avg_price_updates(self, broker):
        await broker.place_order("AAPL", 10, "buy")  # at 150
        broker.set_price("AAPL", 200.0)
        await broker.place_order("AAPL", 10, "buy")  # at 200
        assert broker.positions["AAPL"]["avg_price"] == pytest.approx(175.0)


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_empty(self, broker):
        positions = await broker.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_with_position(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10
        assert positions[0].unrealized_pnl == 0.0

    @pytest.mark.asyncio
    async def test_unrealized_pnl(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        broker.set_price("AAPL", 160.0)
        positions = await broker.get_positions()
        assert positions[0].unrealized_pnl == pytest.approx(100.0)


class TestAccountBalance:
    @pytest.mark.asyncio
    async def test_initial(self, broker):
        balance = await broker.get_account_balance()
        assert balance.equity == 100_000.0
        assert balance.cash == 100_000.0
        assert balance.positions_value == 0.0

    @pytest.mark.asyncio
    async def test_after_buy(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        balance = await broker.get_account_balance()
        assert balance.cash == pytest.approx(98_500.0)
        assert balance.positions_value == pytest.approx(1_500.0)
        assert balance.equity == pytest.approx(100_000.0)

    @pytest.mark.asyncio
    async def test_with_pnl(self, broker):
        await broker.place_order("AAPL", 10, "buy")
        broker.set_price("AAPL", 160.0)
        balance = await broker.get_account_balance()
        assert balance.equity == pytest.approx(100_100.0)


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel_pending(self, broker):
        order = await broker.place_order("AAPL", 10, "buy", order_type="limit", limit_price=140.0)
        assert order.status == "pending"
        result = await broker.cancel_order(order.order_id)
        assert result is True
        updated = await broker.get_order_status(order.order_id)
        assert updated.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_filled_fails(self, broker):
        order = await broker.place_order("AAPL", 10, "buy")
        assert order.status == "filled"
        result = await broker.cancel_order(order.order_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_unknown_fails(self, broker):
        result = await broker.cancel_order("fake-id")
        assert result is False


class TestGetOrderStatus:
    @pytest.mark.asyncio
    async def test_existing_order(self, broker):
        order = await broker.place_order("AAPL", 10, "buy")
        status = await broker.get_order_status(order.order_id)
        assert status.status == "filled"

    @pytest.mark.asyncio
    async def test_unknown_order(self, broker):
        with pytest.raises(ValueError, match="not found"):
            await broker.get_order_status("fake-id")


class TestStreamPrices:
    @pytest.mark.asyncio
    async def test_calls_callback(self, broker):
        received = []

        async def cb(symbol, price):
            received.append((symbol, price))

        await broker.stream_prices(["AAPL", "MSFT"], cb)
        assert ("AAPL", 150.0) in received
        assert ("MSFT", 300.0) in received

    @pytest.mark.asyncio
    async def test_skips_unknown_symbols(self, broker):
        received = []

        async def cb(symbol, price):
            received.append(symbol)

        await broker.stream_prices(["AAPL", "FAKE"], cb)
        assert "AAPL" in received
        assert "FAKE" not in received
