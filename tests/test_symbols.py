"""Tests for src/data/symbols.py — symbol normalization and classification."""

import pytest

from src.data.symbols import (
    CRYPTO_UNIVERSE,
    EQUITY_UNIVERSE,
    FULL_UNIVERSE,
    from_alpaca,
    is_crypto,
    market_type,
    to_alpaca,
)


class TestToAlpaca:
    """Convert internal format to Alpaca API format."""

    def test_crypto_btc(self):
        assert to_alpaca("BTC-USD") == "BTC/USD"

    def test_crypto_eth(self):
        assert to_alpaca("ETH-USD") == "ETH/USD"

    def test_equity_passthrough(self):
        assert to_alpaca("AAPL") == "AAPL"

    def test_equity_spy(self):
        assert to_alpaca("SPY") == "SPY"

    def test_crypto_sol(self):
        assert to_alpaca("SOL-USD") == "SOL/USD"

    def test_crypto_usdt_pair(self):
        assert to_alpaca("BTC-USDT") == "BTC/USDT"


class TestFromAlpaca:
    """Convert Alpaca API format to internal format."""

    def test_crypto_btc(self):
        assert from_alpaca("BTC/USD") == "BTC-USD"

    def test_crypto_eth(self):
        assert from_alpaca("ETH/USD") == "ETH-USD"

    def test_equity_passthrough(self):
        assert from_alpaca("AAPL") == "AAPL"

    def test_roundtrip_crypto(self):
        assert from_alpaca(to_alpaca("BTC-USD")) == "BTC-USD"

    def test_roundtrip_equity(self):
        assert from_alpaca(to_alpaca("MSFT")) == "MSFT"


class TestIsCrypto:
    """Classify symbols as crypto or equity."""

    def test_btc_internal(self):
        assert is_crypto("BTC-USD") is True

    def test_eth_internal(self):
        assert is_crypto("ETH-USD") is True

    def test_btc_alpaca(self):
        assert is_crypto("BTC/USD") is True

    def test_equity(self):
        assert is_crypto("AAPL") is False

    def test_spy(self):
        assert is_crypto("SPY") is False

    def test_sol(self):
        assert is_crypto("SOL-USD") is True

    def test_usdt_pair(self):
        assert is_crypto("BTC-USDT") is True

    def test_case_insensitive(self):
        assert is_crypto("btc-usd") is True

    def test_eur_pair(self):
        assert is_crypto("BTC-EUR") is True


class TestMarketType:
    def test_crypto(self):
        assert market_type("BTC-USD") == "crypto"

    def test_equity(self):
        assert market_type("AAPL") == "equity"


class TestUniverseConstants:
    def test_equity_universe_not_empty(self):
        assert len(EQUITY_UNIVERSE) > 0

    def test_crypto_universe_not_empty(self):
        assert len(CRYPTO_UNIVERSE) > 0

    def test_full_universe_is_combined(self):
        assert FULL_UNIVERSE == EQUITY_UNIVERSE + CRYPTO_UNIVERSE

    def test_no_crypto_in_equity_universe(self):
        for sym in EQUITY_UNIVERSE:
            assert not is_crypto(sym), f"{sym} should not be crypto"

    def test_all_crypto_in_crypto_universe(self):
        for sym in CRYPTO_UNIVERSE:
            assert is_crypto(sym), f"{sym} should be crypto"
