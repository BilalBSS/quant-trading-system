# / tests for broker factory

from __future__ import annotations

import pytest

from src.brokers.broker_factory import BrokerFactory, create_broker
from src.brokers.paper_broker import PaperBroker
from src.brokers.alpaca_broker import AlpacaBroker


class TestBrokerFactory:
    def test_paper_mode(self):
        factory = BrokerFactory(mode="paper")
        assert factory.mode == "paper"
        broker = factory.get_broker("AAPL")
        assert isinstance(broker, PaperBroker)

    def test_paper_mode_crypto(self):
        factory = BrokerFactory(mode="paper")
        broker = factory.get_broker("BTC-USD")
        assert isinstance(broker, PaperBroker)

    def test_live_mode(self):
        factory = BrokerFactory(mode="live")
        assert factory.mode == "live"
        broker = factory.get_broker("AAPL")
        assert isinstance(broker, AlpacaBroker)

    def test_paper_broker_property(self):
        factory = BrokerFactory(mode="paper")
        assert factory.paper_broker is not None
        assert isinstance(factory.paper_broker, PaperBroker)

    def test_paper_broker_none_in_live(self):
        factory = BrokerFactory(mode="live")
        assert factory.paper_broker is None

    def test_initial_cash(self):
        factory = BrokerFactory(mode="paper", initial_cash=50_000.0)
        broker = factory.get_broker()
        assert isinstance(broker, PaperBroker)
        assert broker.cash == 50_000.0


class TestCreateBroker:
    def test_convenience_function(self):
        factory = create_broker(mode="paper", initial_cash=75_000.0)
        assert isinstance(factory, BrokerFactory)
        assert factory.mode == "paper"


# ---------- new deep tests ----------


class TestBrokerFactoryDeep:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="invalid broker mode"):
            BrokerFactory(mode="invalid")

    def test_case_sensitive_mode(self):
        # / "Paper" with capital P should fail
        with pytest.raises(ValueError, match="invalid broker mode"):
            BrokerFactory(mode="Paper")

    def test_get_broker_returns_same_type_consistently(self):
        # / calling get_broker multiple times returns same instance type
        factory = BrokerFactory(mode="paper")
        b1 = factory.get_broker("AAPL")
        b2 = factory.get_broker("MSFT")
        b3 = factory.get_broker("BTC-USD")
        assert type(b1) is type(b2) is type(b3)
        # / in paper mode they should be the exact same instance
        assert b1 is b2 is b3

    def test_mode_property_paper(self):
        factory = BrokerFactory(mode="paper")
        assert factory.mode == "paper"

    def test_mode_property_live(self):
        factory = BrokerFactory(mode="live")
        assert factory.mode == "live"
