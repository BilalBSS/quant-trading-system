"""Tests for src/data/symbols.py — symbol normalization and classification."""

import pytest

from src.data.symbols import (
    CRYPTO_UNIVERSE,
    EQUITY_UNIVERSE,
    FULL_UNIVERSE,
    from_alpaca,
    is_crypto,
    market_type,
    resolve_universe,
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

    def test_gbp_pair(self):
        assert is_crypto("BTC-GBP") is True

    def test_lowercase_with_slash(self):
        assert is_crypto("btc/usd") is True


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


class TestResolveUniverse:
    def test_all_with_available_symbols(self):
        available = ["AAPL", "BTC-USD", "TSLA"]
        result = resolve_universe("all", available_symbols=available)
        assert result == available

    def test_all_without_available_symbols(self):
        result = resolve_universe("all")
        assert result == FULL_UNIVERSE

    def test_all_stocks_filters_out_crypto(self):
        available = ["AAPL", "BTC-USD", "MSFT", "ETH-USD"]
        result = resolve_universe("all_stocks", available_symbols=available)
        assert result == ["AAPL", "MSFT"]

    def test_all_crypto_filters_out_stocks(self):
        available = ["AAPL", "BTC-USD", "MSFT", "ETH-USD"]
        result = resolve_universe("all_crypto", available_symbols=available)
        assert result == ["BTC-USD", "ETH-USD"]

    def test_sp500_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            resolve_universe("sp500")

    def test_nasdaq100_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            resolve_universe("nasdaq100")

    def test_comma_separated_string(self):
        result = resolve_universe("AAPL,MSFT,GOOG")
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_unknown_ref_treated_as_comma_separated(self):
        result = resolve_universe("TSLA")
        assert result == ["TSLA"]

    def test_whitespace_handling_in_comma_separated(self):
        result = resolve_universe(" AAPL , MSFT , GOOG ")
        assert result == ["AAPL", "MSFT", "GOOG"]

    def test_case_insensitive_ref(self):
        result = resolve_universe("ALL")
        assert result == FULL_UNIVERSE

    def test_default_equity_with_available_symbols(self):
        # / "default_equity" has a cached list (EQUITY_UNIVERSE), returns it directly
        available = ["AAPL", "BTC-USD", "TSLA"]
        result = resolve_universe("default_equity", available_symbols=available)
        assert result == EQUITY_UNIVERSE

    def test_crypto_returns_crypto_universe(self):
        result = resolve_universe("crypto")
        assert result == CRYPTO_UNIVERSE


class TestRoundtrip:
    def test_all_full_universe_symbols_roundtrip(self):
        for sym in FULL_UNIVERSE:
            assert from_alpaca(to_alpaca(sym)) == sym, f"{sym} failed roundtrip"
