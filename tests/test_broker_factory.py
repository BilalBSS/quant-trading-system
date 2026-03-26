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
